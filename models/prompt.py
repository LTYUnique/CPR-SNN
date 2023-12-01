import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type
from typing import Sequence, Tuple, Union
from functools import reduce


class Prompts(nn.Module):
    def __init__(
                    self, b, c, N, n_clin_var, num_heads=16, mlp_dim=3072
                ) -> None:


        super().__init__()

        # n_patchs = int(input_size / (patch_size**3))
        # hidden_size = patch_size**3 * channel
        # proj_size = int(hidden_size/proj_rate)

        # print('hidden_size, proj_size :', hidden_size, proj_size)
        self.prompt1 = Prompt(b, c, N, n_clin_var, num_heads, mlp_dim)
        self.prompt2 = Prompt(b, c, N, n_clin_var, num_heads, mlp_dim)
    
    def forward(self, x1, x2, clin_var):
        # print(x1.shape, x2.shape, '/**///*/')
        size = x1.shape
        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)
        
        # print(x1.shape, x2.shape)
        x1, x2 = self.prompt1(x1, clin_var), self.prompt2(x2, clin_var)
        x1, x2 = x1.permute(0, 2, 1).reshape(size), x2.permute(0, 2, 1).reshape(size)
        # print(x1.shape, x2.shape, '/////////////////---------/////////')
        # exit()
        return x1, x2  # x1.permute(0, 2, 1).reshape(size), x2.permute(0, 2, 1).reshape(size)

class Prompts2(nn.Module):
    def __init__(
                    self, c, n_clin_var=8
                ) -> None:


        super().__init__()

        self.EHR_proj1 = nn.Sequential(nn.Linear(n_clin_var, c)) # 8 -> c  b,1,c
        self.EHR_proj2 = nn.Sequential(nn.Linear(n_clin_var, c)) # 8 -> c  b,1,c
    
    def forward(self, x1, x2, clin_var):
        x1 += self.EHR_proj1(clin_var.float().squeeze(dim=1))[:, :, None, None, None]
        x2 += self.EHR_proj2(clin_var.float().squeeze(dim=1))[:, :, None, None, None]
        return x1, x2


class Prompts3(nn.Module):
    def __init__(
                    self, b, c, h, w, d, n_clin_var=8
                ) -> None:


        super().__init__()
        self.desired_size = (b, n_clin_var, d, h, w)
        self.conv1 = nn.Conv3d(c+n_clin_var, c, 3, padding=1)
        self.conv2 = nn.Conv3d(c+n_clin_var, c, 3, padding=1)

        # self.EHR_proj1 = nn.Sequential(nn.Linear(n_clin_var, c)) # 8 -> c  b,1,c
        # self.EHR_proj2 = nn.Sequential(nn.Linear(n_clin_var, c)) # 8 -> c  b,1,c
    
    def forward(self, x1, x2, clin_var):
        clin_var = clin_var.squeeze(dim=1)[:, :, None, None, None].expand(*self.desired_size).float()
        x1 = torch.cat((x1, clin_var), dim=1)
        x2 = torch.cat((x2, clin_var), dim=1)
        
        return self.conv1(x1), self.conv2(x2)



"""
class Prompt(nn.Module):
    def __init__(
                    self,
                    batch_size: int,
                    hidden_size: int,
                    proj_size: int,
                    n_patchs: int,
                    n_clin_var: int = 8,
                    num_heads: int = 8,
                    mlp_dim: int = 3072,
                ):


        super().__init__()

        
        

        self.linear_layer = nn.Linear(hidden_size, proj_size)
        self.revise_linear_layer = nn.Linear(proj_size, hidden_size)

        # add cross-attention
        self.EHR_proj = nn.Sequential(nn.Linear(n_clin_var, proj_size))

        # image positional embedding
        self.image_pe2 = nn.Parameter(torch.zeros(batch_size, n_patchs, proj_size))

        self.ct2 = TwoWayTransformer(depth=1, embedding_dim=hidden_size, num_heads=num_heads, mlp_dim=mlp_dim)
    
    def position_encoding(self, x):
        b, c, h, w, d = x.size()
        patch_size = 8

        # 切分成小体积
        unfolded_tensor = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
        unfolded_tensor = unfolded_tensor.contiguous().view(b, c, -1, patch_size, patch_size, patch_size)

        # # 位置编码
        # position = (
        #     torch.arange(patch_size**2 * patch_size)
        #     .reshape(1, 1, -1, 1, 1, 1)
        #     .to(x.device)
        # )
        # position_enc = position / (10000 ** (2 / (c * patch_size * patch_size * patch_size)))
        # position_enc[:, :, :, :, :, 0::2] = torch.sin(position_enc[:, :, :, :, :, 0::2])
        # position_enc[:, :, :, :, :, 1::2] = torch.cos(position_enc[:, :, :, :, :, 1::2])

        # # 将位置编码应用到切分的小体积
        # encoded_tensor = unfolded_tensor + position_enc

        # # 拉成一维
        # flattened_tensor = encoded_tensor.view(b, c, -1, patch_size**2 * patch_size)
        flattened_tensor = unfolded_tensor.view(b, c, -1, patch_size**2 * patch_size)
        flattened_tensor = flattened_tensor.permute(0, 2, 3, 1).contiguous().view(b, -1, c * patch_size * patch_size * patch_size)
        # b,n_patch,h_*w_*d_*c
        '''
        # 创建输入向量
        x = torch.randn(2, 16, 64, 64, 32)  # 示例输入形状为 (2, 16, 64, 64, 32)

        # 切分、位置编码和拉成一维
        patch_size = 8
        flattened_x = position_encoding(x)

        # 输出形状
        print(flattened_x.size())  # 输出: torch.Size([2, 256, 8192])
        '''
        # print('2', flattened_tensor)
        # exit()
        return self.linear_layer(flattened_tensor)

    def forward(self, x, clin_var):
        B, C, H, W, D = x.size()
        x = self.position_encoding(x)

        prompt_token = self.EHR_proj(clin_var)
        _, x = self.ct2(x, self.image_pe2, prompt_token)

        b, p, n = x.size()
        reshaped_x = x.view(b * p, n)
        output = self.revise_linear_layer(reshaped_x)

        # 调整输出形状
        resized_output = output.view(b, p, -1)  # 调整形状

        # 将形状还原
        return resized_output.view(B, C, H, W, D)  # 调整形状
"""

class Prompt(nn.Module):
    def __init__(self, b, c, N, n_clin_var=8, num_heads=16, mlp_dim=3072):
        super().__init__()       
        # proj_size = int(N/proj_rate)
        print(b, N, c)
        # self.linear_layer = nn.Linear(N, proj_size)
        # self.revise_linear_layer = nn.Linear(proj_size, N)

        # add cross-attention
        # self.EHR_proj = nn.Sequential(nn.Linear(n_clin_var, proj_size))
        self.EHR_proj = nn.Sequential(nn.Linear(n_clin_var, c)) # 8 -> c  b,1,c

        # image positional embedding
        # self.image_pe2 = nn.Parameter(torch.zeros(b, c, proj_size))
        self.image_pe2 = nn.Parameter(torch.zeros(b, N, c))

        self.ct2 = TwoWayTransformer(depth=1, embedding_dim=c, num_heads=num_heads, mlp_dim=mlp_dim)
    

    def forward(self, x, clin_var) ->Tensor:

        prompt_token = self.EHR_proj(clin_var.float()) #  b,1,c
        # x: # b, n c
        _, x = self.ct2(x, self.image_pe2, prompt_token)
        # print('////////////////////////////////////////////////////////////////////////////////////')
        # print(type(x))
        return x # self.revise_linear_layer(x)
    
class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        # bs, c, h, w = image_embedding.shape
        # image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)   # query last

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)     # keys last

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# class TwoWayTransformer(nn.Module):
#     def __init__(
#         self,
#         depth: int,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#     ) -> None:
#         """
#         A transformer decoder that attends to an input image using
#         queries whose positional embedding is supplied.

#         Args:
#           depth (int): number of layers in the transformer
#           embedding_dim (int): the channel dimension for the input embeddings
#           num_heads (int): the number of heads for multihead attention. Must
#             divide embedding_dim
#           mlp_dim (int): the channel dimension internal to the MLP block
#           activation (nn.Module): the activation to use in the MLP block
#         """
#         super().__init__()
#         self.depth = depth
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.mlp_dim = mlp_dim
#         self.layers = nn.ModuleList()

#         for i in range(depth):
#             self.layers.append(
#                 TwoWayAttentionBlock(
#                     embedding_dim=embedding_dim,
#                     num_heads=num_heads,
#                     mlp_dim=mlp_dim,
#                     activation=activation,
#                     attention_downsample_rate=attention_downsample_rate,
#                     skip_first_layer_pe=(i == 0),
#                 )
#             )

#         self.final_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm_final_attn = nn.LayerNorm(embedding_dim)

#     def forward(
#         self,
#         image_embedding: Tensor,
#         image_pe: Tensor,
#         point_embedding: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         """
#         Args:
#           image_embedding (torch.Tensor): image to attend to. Should be shape
#             B x embedding_dim x h x w for any h and w.
#           image_pe (torch.Tensor): the positional encoding to add to the image. Must
#             have the same shape as image_embedding.
#           point_embedding (torch.Tensor): the embedding to add to the query points.
#             Must have shape B x N_points x embedding_dim for any N_points.

#         Returns:
#           torch.Tensor: the processed point_embedding
#           torch.Tensor: the processed image_embedding
#         """
#         # BxCxHxW -> BxHWxC == B x N_image_tokens x C
#         bs, c, h, w = image_embedding.shape
#         image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
#         image_pe = image_pe.flatten(2).permute(0, 2, 1)

#         # Prepare queries
#         queries = point_embedding
#         keys = image_embedding

#         # Apply transformer blocks and final layernorm
#         for layer in self.layers:
#             queries, keys = layer(
#                 queries=queries,
#                 keys=keys,
#                 query_pe=point_embedding,
#                 key_pe=image_pe,
#             )

#         # Apply the final attention layer from the points to the image
#         q = queries + point_embedding
#         k = keys + image_pe
#         attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm_final_attn(queries)

#         return queries, keys


# class TwoWayAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         mlp_dim: int = 2048,
#         activation: Type[nn.Module] = nn.ReLU,
#         attention_downsample_rate: int = 2,
#         skip_first_layer_pe: bool = False,
#     ) -> None:
#         """
#         A transformer block with four layers: (1) self-attention of sparse
#         inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
#         block on sparse inputs, and (4) cross attention of dense inputs to sparse
#         inputs.

#         Arguments:
#           embedding_dim (int): the channel dimension of the embeddings
#           num_heads (int): the number of heads in the attention layers
#           mlp_dim (int): the hidden dimension of the mlp block
#           activation (nn.Module): the activation of the mlp block
#           skip_first_layer_pe (bool): skip the PE on the first layer
#         """
#         super().__init__()
#         self.self_attn = Attention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)

#         self.cross_attn_token_to_image = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )
#         self.norm2 = nn.LayerNorm(embedding_dim)

#         self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
#         self.norm3 = nn.LayerNorm(embedding_dim)

#         self.norm4 = nn.LayerNorm(embedding_dim)
#         self.cross_attn_image_to_token = Attention(
#             embedding_dim, num_heads, downsample_rate=attention_downsample_rate
#         )

#         self.skip_first_layer_pe = skip_first_layer_pe

#     def forward(
#         self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
#     ) -> Tuple[Tensor, Tensor]:
#         # Self attention block
#         if self.skip_first_layer_pe:
#             queries = self.self_attn(q=queries, k=queries, v=queries)
#         else:
#             q = queries + query_pe
#             attn_out = self.self_attn(q=q, k=q, v=queries)
#             queries = queries + attn_out
#         queries = self.norm1(queries)

#         # Cross attention block, tokens attending to image embedding
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
#         queries = queries + attn_out
#         queries = self.norm2(queries)

#         # MLP block
#         mlp_out = self.mlp(queries)
#         queries = queries + mlp_out
#         queries = self.norm3(queries)

#         # Cross attention block, image embedding attending to tokens
#         q = queries + query_pe
#         k = keys + key_pe
#         attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
#         keys = keys + attn_out
#         keys = self.norm4(keys)

#         return queries, keys


# class Attention(nn.Module):
#     """
#     An attention layer that allows for downscaling the size of the embedding
#     after projection to queries, keys, and values.
#     """

#     def __init__(
#         self,
#         embedding_dim: int,
#         num_heads: int,
#         downsample_rate: int = 1,
#     ) -> None:
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.internal_dim = embedding_dim // downsample_rate
#         self.num_heads = num_heads
#         assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

#         self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
#         self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

#     def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
#         b, n, c = x.shape
#         x = x.reshape(b, n, num_heads, c // num_heads)
#         return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

#     def _recombine_heads(self, x: Tensor) -> Tensor:
#         b, n_heads, n_tokens, c_per_head = x.shape
#         x = x.transpose(1, 2)
#         return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

#     def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
#         # Input projections
#         q = self.q_proj(q)
#         k = self.k_proj(k)
#         v = self.v_proj(v)

#         # Separate into heads
#         q = self._separate_heads(q, self.num_heads)
#         k = self._separate_heads(k, self.num_heads)
#         v = self._separate_heads(v, self.num_heads)

#         # Attention
#         _, _, _, c_per_head = q.shape
#         attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
#         attn = attn / math.sqrt(c_per_head)
#         attn = torch.softmax(attn, dim=-1)

#         # Get output
#         out = attn @ v
#         out = self._recombine_heads(out)
#         out = self.out_proj(out)

#         return out


# class MLPBlock(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         mlp_dim: int,
#         act: Type[nn.Module] = nn.GELU,
#     ) -> None:
#         super().__init__()
#         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
#         self.lin2 = nn.Linear(mlp_dim, embedding_dim)
#         self.act = act()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.lin2(self.act(self.lin1(x)))
