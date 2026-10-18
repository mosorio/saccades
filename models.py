# %%
import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group, multivariate_normal
from prettytable import PrettyTable
from modules import RNN, GRU, LSTM, ConvNet, MultRNN, MultiplicativeLayer, SparseLinear
import ventral_models as vmod
from skimage.transform import warp_polar, rotate


def choose_model(config, model_dir):
    # Prepare model arguments
    model_type = config.model_type
    device = config.device
    max_num = config.max_num
    no_symbol = True if config.shape_input == 'parametric' else False
    drop = config.dropout
    n_classes = max_num - config.min_num + 1
    if 'unique' in config.challenge:
        n_classes = 3
    elif 'distract' in config.challenge and config.target_type == 'all':
        n_classes = n_classes + 2

    n_shapes = 2 if config.same and config.sort else 25 # 20 or 25
    if config.ventral is not None:
        ventral = config.ventral
        #ventral = model_dir + '/ventral/' + config.ventral
    else:
        ventral = None
    finetune = True if 'finetune' in model_type else False
    whole_im = True if 'whole' in model_type else False
    height, width = config.height, config.width
    xy_sz = height*width if config.place_code else 2
    sigmoid = False if config.use_loss == 'num' else True
    mod_args = {'h_size': config.h_size, 'act': config.act,
                'detach': config.detach, 'format':config.shape_input,
                'n_classes':n_classes, 'dropout': drop, 'grid': config.grid,
                'n_shapes':n_shapes, 'ventral':ventral, 'train_on':config.train_on,
                'finetune': finetune, 'device':device, 'sort':config.sort,
                'no_pretrain': config.no_pretrain, 'whole':whole_im,
                'n_glimpses': config.n_glimpses, 'xy_sz':xy_sz, 'mult':config.mult, 
                'pass_penult':config.pass_penult, 'sigmoid':sigmoid,
                'task_type': getattr(config, 'task_type', 'count'),
                'map_classes': getattr(config, 'map_shape_count', None),
                'head': config.head, 'min_num': config.min_num, 'max_num': config.max_num,
                'count_mode': config.count_mode,
                'what_divisive_norm': getattr(config, 'what_divisive_norm', 'none'),
                'what_divisive_sigma': getattr(config, 'what_divisive_sigma', 1.0),
                'what_divisive_eps': getattr(config, 'what_divisive_eps', 1e-6),
                'rnn_type': getattr(config, 'rnn_type', 'vanilla'),
                'use_layernorm': getattr(config, 'use_layernorm', False),
                'small_model_encoder': getattr(config, 'small_model_encoder', False),
                'xy_preencode_size': getattr(config, 'xy_preencode_size', 64),
                'joint_hidden_size': getattr(config, 'joint_hidden_size', 256),
                }
    if 'par' in model_type:# == 'rnn_classifier_par':
        # Model with two parallel streams at the level of the map. Only one
        # stream is optimized to match the map. The other of the same size
        # is free, only influenced by the num loss.
        mod_args['parallel'] = True
    hidden_size = mod_args['h_size']
    #output_size = mod_args['n_classes'] #+ 1
    output_size = getattr(config, 'rel_output_size', max_num - config.min_num + 1)
    shape_format = mod_args['format']
    train_on = mod_args['train_on']
    grid = mod_args['grid']
    n_shapes = mod_args['n_shapes']
    head = mod_args['head']
    # grid_to_im_shape = {3:[27, 24], 6:[48, 42], 9:[69, 60]}
    # height, width = grid_to_im_shape[grid]
    map_size = grid**2
    
    if 'onehot' in shape_format:
        sh_sz = n_shapes
    elif 'symbolic' in shape_format:
        sh_sz = n_shapes#20#25
        if config.sort and config.same:
            sh_sz = 2
        else:
            sh_sz = n_shapes#20#25
    elif 'pixel' in shape_format:
        sh_sz = 2 if '+' in shape_format else 1
    else:
        sh_sz = width * height
    if '2channel' in shape_format:
        sh_sz *= 2
    in_sz = xy_sz if train_on=='xy' else sh_sz if train_on =='shape' else sh_sz + xy_sz

    # Initialize the selected model class
    if 'ventral' in model_type:
        model = PretrainedVentral(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'mlp' in model_type:
        in_size = height * width
        model = FeedForward(in_size, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'rnn_classifier' in model_type:
        if '2stream' in model_type:
            model = RNNClassifier2stream(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
        else:
            print(f"Model type '{model_type}' matches 'rnn_classifier' but not a known variant (e.g. 'rnn_classifier2stream'). Exiting")
            exit()
    elif model_type == 'recurrent_control':
        in_sz = height * width
        model = RNNClassifier2stream(in_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'cnn' in model_type:
        dropout = mod_args['dropout']
        if model_type == 'bigcnn':
            mod_args['big'] = True
            model = ConvNet(width, height, map_size, output_size, **mod_args).to(device)
        else:
            model = ConvNet(width, height, map_size, output_size, **mod_args).to(device)
    else:
        print(f'Model type {model_type} not implemented. Exiting')
        exit()
    # if small_weights:
    #     model.init_small()  # not implemented yet
    print('Params to learn:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t {name} {param.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable model params: {total_params}')

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    count_parameters(model)
    return model


class FeedForward(nn.Module):
    def __init__(self, in_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.sig = kwargs['sigmoid']
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, map_size)
        self.layer6 = nn.Linear(map_size, output_size)
        self.drop_layer = nn.Dropout(p=drop)
        self.LReLU = nn.LeakyReLU(0.1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.LReLU(self.layer1(x))
        x = self.LReLU(self.layer2(x))
        x = self.LReLU(self.layer3(x))
        x = self.LReLU(self.layer4(x))
        x = self.drop_layer(x)
        map = self.layer5(x) # want to return the map without any nonlinearity applied because that is what BCEWithLogitsLoss is expecting
        if self.sig:
            map_to_pass_on = self.Sigmoid(map) # Only apply sigmoid if the map loss is being optimised, otherwise it makes learning less stable for no reason, at least in RNNs. Maybe less so here.
        else:
            map_to_pass_on = self.LReLU(map)
        out = self.layer6(map)
        return out, map, x


class PretrainedVentral(nn.Module):
    """Used for the distractor task. Has the main dual-stream module as rnn."""
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.train_on = kwargs['train_on']
        ventral_file = kwargs['ventral']
        self.whole_im = kwargs['whole']
        self.gate = kwargs['gate'] if 'gate' in kwargs.keys() else False
        self.ce = True if 'loss-ce' in ventral_file else False
        self.finetune = kwargs['finetune']
        self.sort = kwargs['sort']
        self.pass_penult =  kwargs['pass_penult']
        no_pretrain = kwargs['no_pretrain']
        penult_size = 10#8
        self.xy_size = kwargs['xy_sz'] if 'xy_sz' in kwargs.keys() else 2

        if self.train_on == 'xy':
            shape_rep_len = 0
        elif self.pass_penult:
            shape_rep_len = penult_size
        else:
            shape_rep_len = 2
            
        if self.finetune:
            drop = 0.4
        else:
            drop = 0
        ventral_output_size = 2
        # output_size = 6
        if 'mlp' in ventral_file :#or 'logpolar' in ventral_file:
            layer_width = 1024
            # self.ventral = vmod.MLP(pix_size, 1024, 3, ventral_output_size, drop=drop)
            self.ventral = vmod.BasicMLP(pix_size, layer_width, penult_size, ventral_output_size, drop=drop)
            self.cnn = False
        elif 'cnn' in ventral_file:
            self.cnn = True
            if self.whole_im or 'logpolar' in ventral_file:
                self.width = 42; self.height = 48
            else:
                self.width = 6; self.height = 6
            self.ventral = vmod.ConvNet(self.width, self.height, penult_size, ventral_output_size, dropout=drop)
        if not no_pretrain:
            print('Loading saved ventral model parameters...')
            map_location = torch.device('cpu')
            try:  # Try loading the whole model first, this is the way to go in case we make changes to the ventral model
                self.ventral = torch.load(ventral_file, map_location=map_location)
                if hasattr(self.ventral, 'penult_size'):
                    assert self.ventral.penult_size == penult_size
            except:  # Otherwise just load the state variables. Assumes the same architecture as initialized above.
                state_dict = torch.load(ventral_file, map_location=map_location)
                self.ventral.load_state_dict(state_dict)

        self.rnn = RNNClassifier2stream(shape_rep_len, hidden_size, map_size, output_size, **kwargs)
        self.n_classes = self.rnn.n_classes
        self.map_classes = getattr(self.rnn, 'map_classes', None)
        self.initHidden = self.rnn.initHidden
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        if self.train_on == 'shape':
            pix = x
        elif self.train_on == 'xy':
            xy = x
        else:
            xy = x[:, :self.xy_size]  # xy coords are first two input features normally, unless place code
            pix = x[:, self.xy_size:]
        if self.cnn and not self.whole_im and self.train_on != 'xy':
            n, p = pix.shape
            pix = pix.view(n, 1, self.height, self.width) # REALLY IMPORTANT THAT THIS IS CORRECT, won't throw error when wrong
        if self.train_on != 'xy':
            if not self.finetune:
                with torch.no_grad():
                    shape_pred, penult_ven = self.ventral(pix) # for BCE experiment
                    if self.pass_penult:
                        shape_rep = penult_ven.detach().clone()
                    else:
                        shape_rep = shape_pred[:, :2].detach().clone()
                        if self.ce:
                            shape_rep = self.Softmax(shape_rep)            

            else:
                shape_pred, penult_ven = self.ventral(pix)
                if self.pass_penult:
                    shape_rep = penult_ven
                else:
                    shape_rep = shape_pred[:, :2]
                    if self.ce:
                            shape_rep = self.Softmax(shape_rep)

            if self.train_on == 'both':
                x = torch.cat((xy, shape_rep), dim=1)
            elif self.train_on == 'shape':
                x = shape_rep
        elif self.train_on == 'xy':
            x = xy
            shape_pred = None
        
        num, pix, map_, hidden, premap, penult, per_shape_scores  = self.rnn(x, hidden)
        return num, shape_pred, map_, hidden, premap, penult, per_shape_scores 

class DivisiveNorm(nn.Module):
    """Divisive normalization on the raw content ('what') stream, applied
    before it enters its embedding layer.

    Per-example evidence is divided by a semi-saturation constant plus the
    total evidence across shapes (`sigma + sum(x)`), so the representation's
    scale doesn't grow with how many shapes have non-zero evidence (e.g. with
    larger K in curriculum-mode scenes). `mode='fixed'` keeps sigma constant;
    `mode='learned'` makes it a trainable scalar.
    """
    def __init__(self, mode='fixed', sigma=1.0, eps=1e-6):
        super().__init__()
        if mode not in ('fixed', 'learned'):
            raise ValueError(f"DivisiveNorm mode must be 'fixed' or 'learned', got {mode!r}")
        self.mode = mode
        self.eps = eps
        if mode == 'learned':
            self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        else:
            self.register_buffer('sigma', torch.tensor(float(sigma)))

    def forward(self, x):
        sigma = self.sigma.clamp(min=self.eps) if self.mode == 'learned' else self.sigma
        denom = (sigma + x.sum(dim=-1, keepdim=True)).clamp(min=self.eps)
        return x / denom


class MapHead(nn.Module):
    """Primary map-prediction head: hidden state -> slot-wise (S+1)-way logits.

    """
    def __init__(self, hidden_dim, n_classes, n_slots):
        super().__init__()
        self.n_classes = n_classes
        self.n_slots = n_slots
        self.map_gen = nn.Linear(hidden_dim, n_classes * n_slots)

    def forward(self, h):
        B = h.size(0)
        return self.map_gen(h).view(B, self.n_classes, self.n_slots)


class CountingHead(nn.Module):
    def __init__(self, hidden_dim, map_dim, n_shapes, n_counts, mode='total'):
        super().__init__()
        self.map_dim  = map_dim
        self.n_shapes = n_shapes
        self.mode     = mode
        self.n_counts = n_counts

        # Generate per-shape map logits from hidden state: [B, S*M]
        self.map_gen  = nn.Linear(hidden_dim, map_dim * n_shapes)

        # Pool a single score from each shape's map (pure linear sum across cells)
        # self.map_pool = nn.Linear(map_dim, 1, bias=False)

        self.total_fc = None
        if mode == 'total':
            assert self.n_counts is not None and self.n_counts > 1, "Need K+1 classes for CE."
            self.total_fc = nn.Linear(map_dim * n_shapes, self.n_counts, bias=True)
        else:
            print("Using per-shape counting head.")
            # Per-shape counting head: each shape's map -> its own K+1 logits
            self.per_shape_fc = nn.Linear(map_dim, self.n_counts, bias=True)
            self.total_fc = None

    def forward(self, h):  # h: [B, hidden_dim]
        B = h.size(0)

        # ----- Per-shape maps (logits for BCE): [B, S, M]
        maps_flat = self.map_gen(h)                         # [B, n_shapes*map_dim]
        per_shape_maps = maps_flat.view(B, self.n_shapes, self.map_dim)  # [B, S, M]

        # ----- Per-shape count logits: [B, S, K+1]
        if self.mode == 'total':
            # total_logits = self.total_fc(per_shape_scores)  # [B, 1]
            #joint_map = per_shape_maps.sum(dim=1)                       # [B, M]
            total_logits = self.total_fc(maps_flat)
            return per_shape_maps, maps_flat, total_logits
            # return per_shape_maps, per_shape_scores, total_logits
        else:
            print("Computing per-shape count logits.")
            per_shape_logits = self.per_shape_fc(
                per_shape_maps.view(B * self.n_shapes, self.map_dim)
            ).view(B, self.n_shapes, self.n_counts)
            return per_shape_maps, None, per_shape_logits
        

class RNNClassifier2stream(nn.Module):
    """Main dual-stream network. 
    
    Used for simple counting as is and as part of PretrainVentral for the
    distractor task.
    Final "number head" produces logits depending on `head`:
        - head='relational' : logits over relational classes (e.g., min/max label space)
        - head='counting'   : per-shape logits over counts 0..K (multi-task classification)
    
    """
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.train_on = kwargs['train_on']
        self.output_size = output_size
        self.n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.xy_size = kwargs['xy_sz'] if 'xy_sz' in kwargs.keys() else 2
        self.mult = kwargs['mult'] if 'mult' in kwargs.keys() else False
        self.sig = kwargs['sigmoid'] if 'sigmoid' in kwargs.keys() else False
        self.grid = kwargs['grid'] if 'grid' in kwargs.keys() else 3
        self.task_type = kwargs['task_type'] if 'task_type' in kwargs.keys() else 'count'
        self.n_classes = kwargs['n_classes'] if 'n_classes' in kwargs.keys() else output_size
        self.map_classes = kwargs['map_classes'] if 'map_classes' in kwargs.keys() else None
        self.min_num = kwargs['min_num'] if 'min_num' in kwargs.keys() else None
        self.max_num = kwargs['max_num'] if 'max_num' in kwargs.keys() else None
        self.map_size = map_size

        # Counting mode: 'total' or 'per_shape'
        self.count_mode = kwargs['count_mode'] if 'count_mode' in kwargs.keys() else 'total'

        self.head = kwargs['head'] if 'head' in kwargs.keys() else None         # 'relational' | 'counting' | 'map'
        self.rnn_type = kwargs.get('rnn_type', 'vanilla')
        if self.rnn_type not in ('vanilla', 'gru', 'lstm'):
            raise ValueError(f"rnn_type must be one of 'vanilla', 'gru', 'lstm', got {self.rnn_type!r}")
        self.use_layernorm = kwargs.get('use_layernorm', False)
        self.small_model_encoder = kwargs.get('small_model_encoder', False)
        self.xy_preencode_size = kwargs.get('xy_preencode_size', 64)
        # Divisive normalization on the raw content ('what') stream, applied
        # before its embedding into the RNN input. 'none' disables it entirely.
        what_divisive_norm = kwargs.get('what_divisive_norm', 'none')
        if what_divisive_norm not in ('none', 'fixed', 'learned'):
            raise ValueError(f"what_divisive_norm must be one of 'none', 'fixed', 'learned', got {what_divisive_norm!r}")
        self.content_norm = None
        if what_divisive_norm != 'none':
            self.content_norm = DivisiveNorm(
                mode=what_divisive_norm,
                sigma=kwargs.get('what_divisive_sigma', 1.0),
                eps=kwargs.get('what_divisive_eps', 1e-6),
            )
        self.count_K = self.max_num   
        self.n_shapes = self.map_classes + 1  
        if self.task_type == 'min':
            self.output_size = int(self.max_num//2) 
        elif self.head == 'counting' and self.count_mode == 'total':
            self.output_size = int(self.max_num - self.min_num) + 1
            # print(self.output_size)
        elif self.head == 'counting' and self.count_mode == 'per_shape':
            self.output_size = int(self.count_K) + 1
            #print('Per-shape counting with n_shapes:', self.n_shapes, 'and count_K:', self.count_K)

        if self.mult:
            embedding_size = 64
        elif self.small_model_encoder and self.train_on == 'both':
            self.xy_preencoder = nn.Linear(self.xy_size, self.xy_preencode_size)
            joint_widths = kwargs.get('joint_hidden_size', None)
            self.joint_hidden_size = list(joint_widths) if joint_widths is not None else [256, hidden_size]
            self.joint_layers = nn.ModuleList()
            in_width = pix_size + self.xy_preencode_size
            for out_width in self.joint_hidden_size:
                self.joint_layers.append(nn.Linear(in_width, out_width))
                in_width = out_width
            embedding_size = in_width  # last fusion layer's width -- decoupled from hidden_size
            for layer in [self.xy_preencoder] + list(self.joint_layers):
                nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        else:
            embedding_size = hidden_size
            self.pix_embedding = nn.Linear(pix_size, embedding_size//2)
            self.xy_embedding = nn.Linear(self.xy_size, embedding_size//2)
        if self.small_model_encoder and self.train_on == 'both':
            pass  # fully encoded by xy_preencoder/joint_layers in forward(); no joint_embedding needed
        elif self.train_on == 'both':
            if self.mult:
                self.joint_embedding = nn.Linear(self.xy_size*pix_size, embedding_size)
            else:
                # pix_embedding + xy_embedding are each embedding_size//2 wide; for odd
                # embedding_size their concatenation is one short of embedding_size itself.
                self.joint_embedding = nn.Linear(2 * (embedding_size // 2), embedding_size)
        else:
            self.joint_embedding = nn.Linear(embedding_size//2, embedding_size)

        print(f'RNNClassifier2stream: train_on {self.train_on}, mult {self.mult}, embedding_size {embedding_size}, pix_size {pix_size}, xy_size {self.xy_size}')

        if self.rnn_type == 'gru':
            self.rnn = GRU(embedding_size, hidden_size, hidden_size, self.act, use_layernorm=self.use_layernorm)
        elif self.rnn_type == 'lstm':
            self.rnn = LSTM(embedding_size, hidden_size, hidden_size, self.act, use_layernorm=self.use_layernorm)
        else:
            self.rnn = RNN(embedding_size, hidden_size, hidden_size, self.act, use_layernorm=self.use_layernorm)
        self.drop_layer = nn.Dropout(p=drop)

        # === Number head(s) ===
        if self.head == 'relational':
            self.map_readout = nn.Linear(hidden_size, map_size)

            #Penultimate (what feeds number head)
            penult_dim = map_size * (2 if self.par else 1)
            if self.par:
                self.notmap = nn.Linear(hidden_size, map_size)
            #logits over min/max classes
            self.num_readout = nn.Linear(penult_dim, self.n_classes, bias=False)
        elif self.head == 'counting':
            # print('Count Mode:', self.count_mode)
            # print('Counting head with n_shapes:', self.n_shapes, 'and output_size:', self.output_size)
            self.count_head = CountingHead(
                hidden_dim=hidden_size,
                map_dim=self.map_size,
                n_shapes=self.n_shapes, 
                n_counts=self.output_size, 
                mode=self.count_mode
            )
        elif self.head == 'map':
            self.map_head = MapHead(hidden_dim=hidden_size, n_classes=self.n_shapes, n_slots=self.map_size)

        # elif self.head == 'counting':
        #     # logits over counts 0..K for each of n_shapes → total units n_shapes*(K+1)
        #     #print("n_shapes",self.n_shapes)
        #     #print("count_K",self.count_K)
        #     out_units = self.n_shapes * (int(self.count_K) + 1)
        #     #print(out_units)
        #     self.num_readout = nn.Linear(penult_dim, out_units, bias=False)
        else:
            raise ValueError(f"Unknown head: {self.head}")

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)


    def forward(self, x, hidden):
        # ----- Stream selection & fusion -----
        if self.train_on == 'both':
            xy = x[:, :self.xy_size]
            pix = x[:, self.xy_size:]
            if self.content_norm is not None:
                pix = self.content_norm(pix)

            if self.small_model_encoder:
                xy_enc = torch.relu(self.xy_preencoder(xy))
                combined = torch.cat((pix, xy_enc), dim=-1)
                for layer in self.joint_layers:
                    combined = torch.relu(layer(combined))
            elif not self.mult:
                xy = self.LReLU(self.xy_embedding(xy))
                pix = self.LReLU(self.pix_embedding(pix))
                combined = torch.cat((xy, pix), dim=-1)
            else:
                combined = torch.einsum('ij,ik->ijk', xy, pix).reshape(-1, xy.shape[1]*pix.shape[1])
        elif self.train_on == 'xy':
            xy = x
            xy = self.LReLU(self.xy_embedding(xy))
            combined = xy
            pix = torch.zeros_like(xy)
        elif self.train_on == 'shape':
            pix = x
            if self.content_norm is not None:
                pix = self.content_norm(pix)
            pix = self.LReLU(self.pix_embedding(pix))
            combined = pix

        # ----- Joint projection → RNN → dropout -----
        if self.small_model_encoder and self.train_on == 'both':
            x = combined  # already fully encoded by xy_preencoder/joint_layers above
        else:
            x = self.LReLU(self.joint_embedding(combined))
        x, hidden = self.rnn(x, hidden)
        x = self.drop_layer(x)

        # # Map path
        # map_ = self.map_readout(x)
        # # If not including the map loss term in the optimized objective function, this sigmoid is unncessary and 
        # # contributes to vanishing gradients. Therefore, when use_loss == num, replace with LReLu.
        # if self.sig:
        #     sig = self.sigmoid(map_)
        # else:
        #     sig = self.LReLU(map_)
        # if self.detach:
        #     map_to_pass_on = torch.round(sig.detach()).clone()
        # else:
        #     map_to_pass_on = sig

        # # penult = self.LReLU(self.after_map(map_to_pass_on))
        # # function fom map to number should be linear so best to omit notlinearity, although because you allready applied sigmoid, lrelu wouldn't do anything anyway
        # # penult = self.after_map(map_to_pass_on) # this extra layer probably isn't helping with anything and just increases the number of params
        # if self.par:
        #     # Two parallel layers, one to be a map, the other not
        #     notmap = self.notmap(x)
        #     penult = torch.cat((map_to_pass_on, notmap), dim=1)
        # else:
        #     penult = map_to_pass_on

        # # Number head forward
        # num_logits_flat = self.num_readout(penult)  # (B, out_units)

        # ---- head forward ----
        if self.head == 'relational':
            # Single flattened map for relational path
            map_ = self.map_readout(x)  # [B, map_size]
            sig = self.sigmoid(map_) if self.sig else self.LReLU(map_)
            map_to_pass_on = torch.round(sig.detach()).clone() if self.detach else sig

            if self.par:
                notmap = self.notmap(x)
                penult = torch.cat((map_to_pass_on, notmap), dim=1)
            else:
                penult = map_to_pass_on

            num_logits_flat = self.num_readout(penult)  # (B, out_units)
            num = num_logits_flat
            per_shape_scores = None

            return num, pix, map_, hidden, x, penult, per_shape_scores

        if self.head == 'map':
            # Pure map objective: no downstream count/relational readout.
            map_ = self.map_head(x)          # [B, n_classes, n_slots]
            penult = map_.reshape(x.size(0), -1)
            return None, pix, map_, hidden, x, penult, None

        # if self.head == 'relational':
        #     num = num_logits_flat

        # else:
            # # Reshape long vector → (B, n_shapes, K+1) so each shape has its own (K+1)-way logits.
            # B = num_logits_flat.size(0)
            # K1 = int(self.count_K) + 1
            # num = num_logits_flat.view(B, self.n_shapes, K1)
            # #print("num", num.shape)
        
        else:
            # Counting path: generate per-shape maps inside the head
            per_shape_maps, per_shape_scores, num = self.count_head(x)  # maps: [B,S,M]
            B = x.size(0)
            map_   = per_shape_maps                     # for BCEWithLogitsLoss (raw logits)
            #print("per_shape_maps", per_shape_maps.shape)
            penult = per_shape_maps.view(B, -1)         # flattened per-shape maps 

            return num, pix, map_, hidden, x, penult, per_shape_scores



        # return num, pix, map_, hidden, x, penult
