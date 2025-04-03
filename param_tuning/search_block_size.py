precisions = [
    # (8, 8),
    (8, 4),
    (8, 2),
    # (4, 4),
    (4, 2),
    (4, 1),
]

HEAD_SIZE = 128
ALIGNMENT = 32

# each vec takes 16 bits
K_VEC_SIZE = 8
V_VEC_SIZE = 4
VEC_BYTES = 2

''' Cache block layout (multiples of 32 bytes)
NUM_PACKS = HEAD_SIZE * BITS / 16
# Key (data) -- padded to 32 bytes
  shape = [NUM_PACKS/VEC_SIZE, NUM_TOKENS, VEC_SIZE] in INT16
  NOTE: make sure NUM_PACKS % VEC_SIZE == 0
# Key (scale_factor + zero_point) -- padded to 32 bytes
# Value (data) -- padded to 32 bytes
  shape = [NUM_TOKENS/VEC_SIZE, NUM_PACKS, VEC_SIZE] in INT16
  NOTE: padded is required in the 0th dimension
# Value (scale_factor + zero_point) -- padded to 32 bytes
# Metadata (score + index)
'''

def divide_round_up(x, y):
    return (x + y - 1) // y

# effective memory used by each token
def compute_token_bytes(kbits: int, vbits: int) -> int:
    global HEAD_SIZE
    
    assert HEAD_SIZE % 8 == 0
    
    k_bytes = kbits * HEAD_SIZE // 8
    v_bytes = vbits * HEAD_SIZE // 8
    
    quant_bytes = 4 # scaling & zero_point, both in FP16
    meta_bytes = 8 # index & score, in int32 & FP32
    
    return k_bytes, v_bytes, quant_bytes, meta_bytes


def block_bytes_from_size(
    block_size: int, 
    kbits: int, 
    vbits: int, 
) -> int:
    global ALIGNMENT
    
    # each pack is 16 bits (INT16)
    k_pack_size = 16 // kbits
    v_pack_size = 16 // vbits
    
    k_num_packs = HEAD_SIZE // k_pack_size
    v_num_packs = HEAD_SIZE // v_pack_size
    
    # NOTE: key shape = [NUM_PACKS/VEC_SIZE, NUM_TOKENS, VEC_SIZE]
    # make sure that keys do not need padding in the 1st dimension
    assert k_num_packs % K_VEC_SIZE == 0
    
    # effective data size
    k_bytes, v_bytes, quant_bytes, meta_bytes = compute_token_bytes(kbits, vbits)
    
    # data (padded) + quant_meta (padded)
    sum_kbytes = divide_round_up(block_size * k_bytes, ALIGNMENT) * ALIGNMENT + \
                 divide_round_up(block_size * quant_bytes, ALIGNMENT) * ALIGNMENT
    
    # NOTE: val shape [NUM_TOKENS/VEC_SIZE, NUM_PACKS, VEC_SIZE]
    padded_v_vecs = divide_round_up(block_size, V_VEC_SIZE)
    padded_v_vec_bytes = padded_v_vecs * v_num_packs * V_VEC_SIZE * VEC_BYTES 
                 
    # data (padded) + quant_meta (padded)
    sum_vbytes = divide_round_up(padded_v_vec_bytes, ALIGNMENT) * ALIGNMENT + \
                 divide_round_up(block_size * quant_bytes, ALIGNMENT) * ALIGNMENT
    
    sum_meta_bytes = divide_round_up(block_size * meta_bytes, ALIGNMENT) * ALIGNMENT
    
    sum_bytes = sum_kbytes + sum_vbytes + sum_meta_bytes
    assert sum_bytes % ALIGNMENT == 0
    return sum_bytes


def block_size_from_bytes(
    block_bytes: int, 
    kbits: int, 
    vbits: int,
) -> int:
    global ALIGNMENT
    
    assert block_bytes % ALIGNMENT == 0
    k_bytes, v_bytes, quant_bytes, meta_bytes = compute_token_bytes(kbits, vbits)
    
    # there are 5 padded regions, so in the worse in case we need to pad 5 ALIGNMENT bytes
    block_size = block_bytes // (k_bytes + v_bytes + 2 * quant_bytes + meta_bytes)
    while block_size > 0:
        padded_block_bytes = block_bytes_from_size(block_size, kbits, vbits)
        # print(f'block_bytes = {block_bytes}, block_size = {block_size}, padded_block_bytes = {padded_block_bytes}')
        if padded_block_bytes <= block_bytes:
            break
        block_size -= 1
    assert block_size > 0, block_size
    return block_size

def get_residual_ratio(
    block_bytes: int,
    block_size: int,
    kbits: int,
    vbits: int
) -> float:
    k_bytes, v_bytes, quant_bytes, meta_bytes = compute_token_bytes(kbits, vbits)
    total_bytes = block_size * (k_bytes + v_bytes + 2 * quant_bytes + meta_bytes)
    return 1 - total_bytes / block_bytes

# Assume that each token is not padded

start = 800
end = 4000

# print(get_residual(104))

min_ratio = 500
min_ratio_pages = []

for b in range(start // ALIGNMENT, end // ALIGNMENT + 1):
    _block_bytes = b * ALIGNMENT
    print(f'** block_bytes = {_block_bytes}')
    res_ratio = 0
    for (kbits, vbits) in precisions:
        _block_size = block_size_from_bytes(_block_bytes, kbits, vbits)
        _r = get_residual_ratio(_block_bytes, _block_size, kbits, vbits)
        print(f'****** k{kbits}v{vbits}, block_size = {_block_size}, res_ratio = {_r}, '
              f'token_bytes = {compute_token_bytes(kbits, vbits)}')
        res_ratio = max(_r, res_ratio)
    if min_ratio > res_ratio:
        min_ratio = res_ratio
        min_ratio_pages = [_block_bytes]
    elif min_ratio == res_ratio:
        min_ratio_pages.append(_block_bytes)

print(f'min residual ratio')
print(min_ratio)
print(min_ratio_pages)


