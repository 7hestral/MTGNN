from layer import *
from torch import nn

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.latent_embedding_size = 128
        self.decoder_hidden_size = 512
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.decoder_filter_convs = nn.ModuleList()
        self.decoder_gate_convs = nn.ModuleList()
        self.decoder_gconv1 = nn.ModuleList()
        self.decoder_gconv2 = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()

        self.mu_transform = nn.Linear(residual_channels * num_nodes, residual_channels * num_nodes)
        self.logvar_transform = nn.Linear(residual_channels * num_nodes, residual_channels * num_nodes)

        pooling_kernel_size = 7
        last_dim = 2**layers
        self.pooling = nn.MaxPool2d(kernel_size=(1, pooling_kernel_size))
        intermediate_channel = ((last_dim - (pooling_kernel_size - 1) - 1) // pooling_kernel_size + 1) * residual_channels
        self.end_conv_1 = nn.Conv2d(in_channels=intermediate_channel,
                                     out_channels=intermediate_channel//2,
                                     kernel_size=(1,1),
                                     bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=intermediate_channel // 2,
                                  out_channels=out_dim,
                                  kernel_size=(1, 1),
                                  bias=True)


        # self.decoder_linear1 = nn.Linear(self.latent_embedding_size, self.decoder_hidden_size)
        # self.decoder_linear2 = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
        # self.decoder_linear3 = nn.Linear(self.decoder_hidden_size, out_dim * num_nodes)



        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                # self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                #                                     out_channels=residual_channels,
                #                                  kernel_size=(1, 1)))
                # if self.seq_length>self.receptive_field:
                #     self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                #                                     out_channels=skip_channels,
                #                                     kernel_size=(1, self.seq_length-rf_size_j+1)))
                # else:
                #     self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                #                                     out_channels=skip_channels,
                #                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.decoder_filter_convs.append(dilated_deconv(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.decoder_gate_convs.append(dilated_deconv(residual_channels, conv_channels, dilation_factor=new_dilation))

                if self.gcn_true:
                    self.decoder_gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.decoder_gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.decoder_norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.decoder_norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        # self.end_conv_1 = nn.Conv2d(in_channels=residual_channels,
        #                                      out_channels=end_channels,
        #                                      kernel_size=(1,1),
        #                                      bias=True)
        # self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
        #                                      out_channels=out_dim,
        #                                      kernel_size=(1,1),
        #                                      bias=True)
        # if self.seq_length > self.receptive_field:
        #     self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
        #     self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        #
        # else:
        #     self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
        #     self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        print('input.shape:', input.shape)


        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        print('start_conv(input).shape:', x.shape)
        # skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            print(i, 'round')
            # residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            print('after time dilation', x.shape)
            # s = x
            # s = self.skip_convs[i](s)
            # skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)
            print('after gconv', x.shape)
            # x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
            print('after norm', x.shape)
        # skip = self.skipE(x)#  + skip
        # only operate on idx
        # if idx is None:
        #     x = x[:, :, idx, :]
        # else:
        #     x = x[:, :, idx, :]
        x = x.view(x.shape[0], -1)
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)

        # reparametrization
        x = self.reparameterize(mu, logvar)
        # decode (TODO: use another )
        x = x.view(x.shape[0], -1, self.num_nodes, 1)
        for i in range(self.layers):
            print(i, 'round')
            # residual = x
            filter = self.decoder_filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.decoder_gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            print('after time deconv', x.shape)
            if self.gcn_true:
                x = self.decoder_gconv1[i](x, adp) + self.decoder_gconv2[i](x, adp.transpose(1, 0))
            print('after gconv', x.shape)
            if idx is None:
                x = self.decoder_norm[i](x, self.idx)
            else:
                x = self.decoder_norm[i](x, idx)
            print('after decoder norm', x.shape)

        # x = x.transpose(1,2)
        # print('after transpose', x.shape)
        # x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = x.unsqueeze(1)
        x = self.pooling(x)
        print('after pooling', x.shape)
        x = x.transpose(2,3)
        x = x.reshape(x.shape[0], -1, self.num_nodes, 1)

        # x = self.decoder_linear1(x)
        # x = F.relu(self.decoder_linear2(x))
        # x = self.decoder_linear3(x)
        # print('skipE(x)', skip.shape)
        # x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # x = x.view(x.shape[0], self.num_nodes, -1)
        print("final x shape", x.shape)
        return x, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            # Get standard deviation
            std = torch.exp(logvar)
            # Returns random numbers from a normal distribution
            eps = torch.randn_like(std)
            # Return sampled values
            return eps.mul(std).add_(mu)
        else:
            return mu