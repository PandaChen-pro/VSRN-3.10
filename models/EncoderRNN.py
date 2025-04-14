import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'): # Keep default as bool for clarity
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        # --- 修改开始 ---
        # 显式将传入的 bidirectional 转换为布尔类型
        self.bidirectional = bool(bidirectional)
        # --- 修改结束 ---
        self.rnn_cell = rnn_cell
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 特征转换层
        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)
        rnn_class = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=dim_hidden,
            hidden_size=dim_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=self.bidirectional, # <-- 现在传递的是正确的布尔类型
            dropout=rnn_dropout_p if n_layers > 1 else 0
        )
        # 初始化权重
        nn.init.xavier_normal_(self.vid2hid.weight)
        nn.init.zeros_(self.vid2hid.bias)
        self.to(self.device) # 确保整个模块移动到设备
    def forward(self, vid_feats):
        vid_feats = vid_feats.to(self.device) # 确保输入在正确的设备上
        batch_size, seq_len, dim_vid = vid_feats.size()
        # 特征转换
        vid_feats_reshaped = self.vid2hid(vid_feats.reshape(-1, dim_vid))
        vid_feats_dropped = self.input_dropout(vid_feats_reshaped)
        vid_feats_rnn_input = vid_feats_dropped.reshape(batch_size, seq_len, self.dim_hidden)
        # 初始化隐藏状态
        h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1),
                         batch_size, self.dim_hidden, device=self.device)
        # 根据RNN类型调用
        # 确保输入和隐藏状态是连续的，有时有助于解决底层问题
        vid_feats_rnn_input = vid_feats_rnn_input.contiguous()
        h0 = h0.contiguous()
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros_like(h0, device=self.device).contiguous() # 确保c0也在设备上且连续
            output, (hidden, _) = self.rnn(vid_feats_rnn_input, (h0, c0))
        else: # GRU case
            output, hidden = self.rnn(vid_feats_rnn_input, h0) # h0 已经是连续的
        return output, hidden


if __name__ == '__main__':
    # Simple test
    encoder = EncoderRNN(dim_vid=2048, dim_hidden=512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vid_feats = torch.randn(32, 10, 2048).to(device)  # batch_size=32, seq_len=10, dim_vid=2048
    output, hidden = encoder(vid_feats)
    print(f"Output shape: {output.shape}")  # Expected: [32, 10, 512]
    print(f"Hidden shape: {hidden.shape}")  # Expected: [1, 32, 512] for GRU