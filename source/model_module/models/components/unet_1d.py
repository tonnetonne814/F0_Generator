import torch
from source.model_module.diffusion.diffusers_1d_modules import ResBlock1D, Downsample1D, Upsample1D, OutConv1DBlock
from source.model_module.models.components.attentions import Encoder as SelfAttn
from source.model_module.models.components.attentions import Decoder as CrossAttn # 正しいか不明

class U_Net_1D(torch.nn.Module):
    def __init__(self,in_ch=192,             # Hidden
                    inner_ch=256,          #
                    filter_ch=768,
                    out_ch=4,              # f0, mask, v, uv
                    time_embed_dim=512,
                    attn_filter_ch=768,
                    ):
        super().__init__()

        ########################
        ### For Down Process ###
        ########################
        self.ResBlock_D1 = ResBlock1D(inp_channels=in_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_D1 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        self.DownSample_D1 =Downsample1D(channels=inner_ch,
                                         use_conv=True,
                                         out_channels=inner_ch)

        self.ResBlock_D2 = ResBlock1D(inp_channels=inner_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_D2 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        self.DownSample_D2 =Downsample1D(channels=inner_ch,
                                         use_conv=True,
                                         out_channels=inner_ch)

        #######################
        ### For Mid Process ###
        #######################
        self.ResBlock_M1 = ResBlock1D(inp_channels=inner_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.SelfAttn    = SelfAttn(hidden_channels=inner_ch,
                                    filter_channels=filter_ch,
                                    n_heads=4,
                                    n_layers=1,
                                    kernel_size=3,
                                    p_dropout=0.1)
        self.ResBlock_M2 = ResBlock1D(inp_channels=inner_ch,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)

        ######################
        ### For Up Process ###
        ######################
        self.UpSample_U1 = Upsample1D(channels=inner_ch,
                                      use_conv=True,
                                      out_channels=inner_ch)
        self.ResBlock_U1 = ResBlock1D(inp_channels=inner_ch*2,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_U1 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)

        self.UpSample_U2 = Upsample1D(channels=inner_ch,
                                      use_conv=True,
                                      out_channels=inner_ch)
        self.ResBlock_U2 = ResBlock1D(inp_channels=in_ch*2,
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_U2 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)

        self.final = OutConv1DBlock(num_groups_out=8,
                                    out_channels=out_ch,
                                    embed_dim=inner_ch,
                                    act_fn="mish")

    def forward(self, x, x_mask, h, h_mask, t):

        # Down Process
        x = self.ResBlock_D1(inputs=x, t=t)
        x_0 = self.CrossAttn_D1(x=x, x_mask=x_mask, h=h, h_mask=h_mask)
        _, _, x_0_len = x_0.shape
        x = self.DownSample_D1(inputs=x_0) # length/2

        x = self.ResBlock_D2(inputs=x, t=t)
        x_mask_1 = self.mask_downsample(downsample_x=x, mask=x_mask)
        x_1 = self.CrossAttn_D2(x=x, x_mask=x_mask_1, h=h, h_mask=h_mask)
        _, _, x_1_len = x_1.shape
        x = self.DownSample_D2(inputs=x_1) # length/4

        # Mid Process
        x = self.ResBlock_M1(inputs=x, t=t)
        x_mask_2 = self.mask_downsample(downsample_x=x, mask=x_mask_1)
        x = self.SelfAttn(x=x,x_mask=x_mask_2)
        x = self.ResBlock_M2(inputs=x, t=t) # length/4

        # Up Process
        x = self.UpSample_U1(inputs=x) # length/2
        x = torch.cat(tensors=[x_1, x[:,:,:x_1_len]], dim=1) # concat in "channel direction"
        x = self.ResBlock_U1(inputs=x, t=t)
        x = self.CrossAttn_U1(x=x, x_mask=x_mask_1, h=h, h_mask=h_mask)

        x = self.UpSample_U2(inputs=x)
        x = torch.cat(tensors=[x_0, x[:,:,:x_0_len]], dim=1) # concat in "channel direction"
        x = self.ResBlock_U2(inputs=x, t=t)
        x = self.CrossAttn_U2(x=x, x_mask=x_mask, h=h, h_mask=h_mask)

        # Final Process
        x = self.final(x)
        return x

    def mask_downsample(self, downsample_x, mask):
        _,_,mask_length = mask.shape
        _,_,x_length = downsample_x.shape
        if mask_length % 2 == 0: # even
            mask = mask[:,:,1::2]
        else:
            mask = mask[:,:,::2]

        if x_length == mask.size(2):
            return mask

def swish(x):
    return x * torch.sigmoid(x)
