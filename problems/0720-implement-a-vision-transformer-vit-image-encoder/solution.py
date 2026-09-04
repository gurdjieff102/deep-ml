def vit_encode(image, patch_size: int, W_patch, cls_token, pos_embed, W_qkv, W_o, num_heads: int):

    image = np.asarray(image)
    W_patch = np.asarray(W_patch)
    cls_token = np.asarray(cls_token)
    pos_embed = np.asarray(pos_embed)
    W_qkv = np.asarray(W_qkv)
    W_o = np.asarray(W_o)

    H, W, C = image.shape
    D = W_patch.shape[1]

    # 1. Split image into patches
    patches = []

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[
                i:i + patch_size,
                j:j + patch_size,
                :
            ]

            patch = patch.reshape(-1)
            patches.append(patch)

    patches = np.array(patches)

    # 2. Patch embedding
    patch_embeddings = patches @ W_patch

    # 3. Add CLS token
    tokens = np.vstack([
        cls_token,
        patch_embeddings
    ])

    # 4. Add positional embedding
    tokens = tokens + pos_embed

    # 5. QKV projection
    qkv = tokens @ W_qkv

    Q = qkv[:, :D]
    K = qkv[:, D:2 * D]
    V = qkv[:, 2 * D:]

    # 6. Split heads
    head_dim = D // num_heads
    seq_len = tokens.shape[0]

    Q = Q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)

    # 7. Attention
    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(head_dim)

    # Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)

    attention = np.exp(scores)
    attention = attention / np.sum(
        attention,
        axis=-1,
        keepdims=True
    )

    # 8. Attention output
    context = attention @ V

    # Concatenate heads
    context = context.transpose(1, 0, 2)
    context = context.reshape(seq_len, D)

    # 9. Output projection
    output = context @ W_o

    # 10. CLS token
    return output[0].tolist()