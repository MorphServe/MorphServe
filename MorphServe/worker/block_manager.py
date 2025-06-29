import torch

from .kernels.block_mgmt import set_block_table_and_num_seq_alloc_blocks, unset_block_table_and_num_seq_alloc_blocks, gather_allocated_blocks_and_unset

class BlockManager:
    """
    BlockManager - Manage the block table and free blocks on CPU / GPU

    This manager records the mapping from (sequence ID, block index) to block 
    ID (which we call `block_table`), and provides methods to allocate and free
    blocks.

    All tables (the block table, the `num_seq_allocated_blocks`, and the free block
    list) are all maintained on the GPU, so that we can leverage custom Triton
    kernels for fast operations.
    """

    def __init__(self, device_name: str, num_blocks: int, max_seqs_in_block_table: int, max_blocks_per_seq: int, block_size: int):
        self.device_name = device_name
        self.num_free_blocks = num_blocks
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        self.num_free_blocks_org = num_blocks
        self.num_blocks_org = num_blocks

        # seq_id |-> number of blocks allocated for this sequence
        # self.num_seq_allocated_blocks = [0, 0, 0, ..., 0]    # 128 entries mean max_seqs_in_block_table=128
        # self.num_seq_allocated_blocks = [3, 0, 0, ..., 0]    # If sequence 1 requires 3 blocks
        # shape = (128)
        self.num_seq_allocated_blocks = torch.zeros(
            (max_seqs_in_block_table,),
            dtype=torch.int32,
            device="cuda"
        )

        # (seq_id, block_index) |-> block_id
        # The block_table maps each sequence to its assigned block IDs
        # self.block_table = [ [5, 6, 7, -1, -1, -1, ..., -1], [-1, -1, -1, ..., -1], ...] # If sequence 0 receives blocks 5, 6, and 7
        # shape = (128, 3072)
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        self.block_table.fill_(-1)

        # block_id |-> whether this block is free or not
        # self.is_block_free = [False, False, False, True, True, True, True, True] # Allocating Blocks 0, 1, and 2
        # shape = (926)
        self.is_block_free = torch.ones(
            (num_blocks,),
            dtype=torch.bool,
            device="cuda"
        )
    
    def get_num_used_blocks(self) -> int:
        """
        Get the number of blocks currently in use.
        This is calculated as the total number of blocks minus the number of free blocks.
        """
        return self.num_blocks - self.num_free_blocks

    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate the requested number of blocks, update relevant status, and
        return the block IDs.
        """
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"No enough free blocks available on {self.device_name} ({self.num_blocks} in total, {self.num_free_blocks} free, {num_blocks} requested)")
        
        # Get the first num_blocks free blocks
        # for example, if self.is_block_free = [False, True, False, False, True, True, ...]
        # selected_blocks = [1, 4] for the first num_blocks = 2 free blocks
        selected_blocks = torch.nonzero(self.is_block_free)[:num_blocks].view(-1)
        self.num_free_blocks -= num_blocks
        self.is_block_free[selected_blocks] = False
        return selected_blocks
    
    def _free_blocks(self, block_ids: torch.Tensor):
        """
        Free the specified blocks, and update relevant status.
        """
        self.num_free_blocks += len(block_ids)
        self.is_block_free[block_ids] = True
    
    def allocate_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """
        Allocate blocks for sequences, making sure that seq #i has at least 
        ceil(target_lengths[i] / block_size) blocks allocated.

        Return new blocks allocated for the sequences. (useful for swapping)
        """
        target_num_blocks = (target_lens + (self.block_size-1)) // self.block_size
        assert (self.num_seq_allocated_blocks[seq_ids] <= target_num_blocks).all(), \
            f"""(On {self.device_name}) Logic error: Some sequences have more blocks already allocated than needed.
                seq_ids: {seq_ids}, target_lens: {target_lens}, target_num_blocks: {target_num_blocks},
                self.num_seq_allocated_blocks[seq_ids]: {self.num_seq_allocated_blocks[seq_ids]}"""
        block_needed = target_num_blocks - self.num_seq_allocated_blocks[seq_ids]
        n_needed = int(torch.sum(block_needed).item())
        
        if n_needed == 0:
            return torch.empty(0, device="cuda:0", dtype=torch.int64)
        
        new_blocks = self._allocate_blocks(torch.sum(block_needed))
        
        # if max(new_blocks) >= self.num_blocks_org:
            # print(f"[BlockManager With New KV Blocks] max(new_blocks) >= self.num_blocks_org: {max(new_blocks)} >= {self.num_blocks_org}, with new_blocks: {new_blocks}, seq_ids: {seq_ids}, block_needed: {block_needed}")
        set_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, new_blocks, seq_ids, block_needed)
        # if max(new_blocks) >= self.num_blocks_org:
            # print(f"~~~~~ after allocate_blocks_for_seqs, max(new_blocks) >= self.num_blocks_org: {max(new_blocks)} >= {self.num_blocks_org}, with new_blocks: {new_blocks}, seq_ids: {seq_ids}, block_needed: {block_needed}")
        return new_blocks
        
    def free_blocks_for_seqs(self, seq_ids: torch.Tensor):
        """
        Free blocks for sequences.
        """
        self.num_free_blocks += torch.sum(self.num_seq_allocated_blocks[seq_ids])
        unset_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)

    def gather_allocated_blocks_and_free(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Gather the block IDs allocated for the specified sequences and mark them as free

        Useful fow swapping in/out
        """
        gathered_block_ids = gather_allocated_blocks_and_unset(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)
        self.num_free_blocks += len(gathered_block_ids)
        return gathered_block_ids

    def get_num_allocated_blocks(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the number of blocks allocated for the specified sequences
        Useful for swapping
        """
        return self.num_seq_allocated_blocks[seq_ids]
    