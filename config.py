"""Default simulation parameters can be controlled via the command line arguments below."""
import argparse
import hashlib


def get_base_name(config):
    """Prepare base file name for results files to be saved."""
    model_type = config.model_type
    target_type = config.target_type
    train_on = config.train_on
    noise_level = config.noise_level
    train_size = config.curriculum_train_size if getattr(config, 'curriculum', '') else config.train_size
    n_epochs = config.n_epochs
    min_num = config.min_num
    max_num = config.max_num
    use_loss = config.use_loss
    drop = config.dropout

    # 
    n_glimpses = f'{config.n_glimpses}_' if config.n_glimpses is not None else ''
    detach = '-detach' if config.detach else ''
    pretrain = '-nopretrain' if config.no_pretrain else ''

    # same = 'same' if config.same else ''
    if config.same or config.distinctive==0 and not config.mixed:
        shape_distinctiveness = 'same'
    elif config.mixed:
        shape_distinctiveness = 'mixed'
    else:
        shape_distinctiveness = ''
    challenge = config.challenge
    curriculum_tag = f'_curr-{config.curriculum}' if getattr(config, 'curriculum', '') else ''
    # solar = 'solarized_' if config.solarize else ''
    transform = f'{config.shape_input}_' if 'logpolar' in config.shape_input else 'gw6_'
    if config.transform:
        transform += '-rotated'
    shapes = ''.join([str(i) for i in config.shapestr])
    sort = 'sort_' if config.sort else ''
    policy = config.policy
    misalign = 'misalign_' if config.misalign else ''
    
    rnn_type = getattr(config, 'rnn_type', 'vanilla')
    rnn_tag = f'_rnn-{rnn_type}' + ('-ln' if getattr(config, 'use_layernorm', False) else '')
    what_divisive_norm = getattr(config, 'what_divisive_norm', 'none')
    wdn_tag = f'_wdn-{what_divisive_norm}' if what_divisive_norm != 'none' else ''
    class_weights_type = getattr(config, 'class_weights_type', 'uniform')
    cw_tag = f'_cwt-{class_weights_type}'
    if class_weights_type == 'dataset_based':
        cw_tag += f'-{getattr(config, "abs_weight_fn", "sqrt_inv")}'

    if getattr(config, 'small_model_encoder', False):
        # Fixed-length hash instead of spelling out xy_preencode_size/joint_hidden_size --
        # a literal tag grows with however many layers joint_hidden_size has, which risks
        # exceeding the OS filename length limit on already-long curriculum filenames.
        enc_key = f'{config.xy_preencode_size}-{"x".join(str(w) for w in config.joint_hidden_size)}'
        sme_tag = f'_sme-{hashlib.md5(enc_key.encode()).hexdigest()[:6]}'
    else:
        sme_tag = ''

    # model_desc = f'{model_type}{detach}{act}{pretrain}_hsize-{config.h_size}_input-{train_on}{kernel}_{config.shape_input}'
    model_desc = f'{model_type}{detach}{pretrain}_hsize-{config.h_size}_input-{train_on}_{config.shape_input}{rnn_tag}{wdn_tag}{sme_tag}'
    # data_desc = f'num{min_num}-{max_num}_nl-{noise_level}_grid{config.grid}_policy-{policy}_trainshapes-{shapes}{same}_{challenge}_{transform}{n_glimpses}{train_size}'
    data_desc = f'num{min_num}-{max_num}_nl-{noise_level}_policy-{policy}_trainshapes-{shapes}{shape_distinctiveness}_{challenge}{curriculum_tag}_{transform}{n_glimpses}{misalign}{train_size}'
    # train_desc = f'loss-{use_loss}_niters-{n_iters}_{n_epochs}eps'
    withshape = '+shape' if config.learn_shape else ''
    lr_tag = '-flatlr' if getattr(config, 'lr_schedule', 'step') == 'flat' else ''
    train_desc = f'loss-{use_loss}{withshape}{cw_tag}_opt-{config.opt}{lr_tag}_drop{drop}_{sort}count-{target_type}_{n_epochs}eps_rep{config.rep}'
    base_name = f'{model_desc}_{data_desc}_{train_desc}'
    # if config.small_weights:
    #     base_name += '_small'
    return base_name


def get_config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    # Params that determine what datasets to load
    parser.add_argument('--noise_level', type=float, default=0.74)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--grid', type=int, default=6)
    parser.add_argument('--same', action='store_true', default=False, help='whether all target characters within the image are the same')
    parser.add_argument('--mixed', action='store_true', default=False, help='whether to train on a mixture of images whose items are all the same or all distinctive')
    # The above two params are replaced with one param --distinctive which can also represent graded levels of distinctiveness, but perhaps we leave the above two for backwards compatibility
    parser.add_argument('--distinctive', type=float, default=0, help='How distinctive should items within a single image be? 0 means all the same shape, 1 means as distinctive as possible, 0.3 and 0.6 in between.')
    parser.add_argument('--challenge', type=str, default='', help='Which images/task to train on. ignore012, ignore123, or "" for simple counting (no special challenge)')
    parser.add_argument('--task_type', type=str, default='count')
    parser.add_argument('--head', type=str, default='relational', choices=['relational', 'counting', 'map'],
        help="'map' is the primary slot-wise (S+1)-way map-prediction head, with no downstream count/relational readout attached.")
    parser.add_argument('--count_mode', type=str, default='total', choices=['total', 'per_shape'], help='How the counting head aggregates maps.')
    parser.add_argument('--map_mode', type=str, default='ce', choices=['bce', 'ce', 'both', 'fixed_absence'], help='Loss function to use for map prediction.')
    parser.add_argument('--bce_weight', type=float, default=1.0, help='Weight for BCE map loss when map_mode is both or bce_then_ce.')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Weight for CE map loss when map_mode is both or bce_then_ce.')
    parser.add_argument('--pos_weight', action='store_true', default=False, help='Whether to use positive weighting for the BCE map loss.')
    parser.add_argument('--map_bce_pretrain_epochs', type=int, default=0, help='Epoch count for BCE-only warmup when map_mode=bce_then_ce.')
    parser.add_argument('--class_weights_type', type=str, default='uniform', choices=['uniform', 'dataset_based'],
        help="Absence-class weighting for the CE map loss (map_mode in {'ce','both'}). "
             "'dataset_based' computes the weight once from the training set's aggregate occupied/absence frequency.")
    parser.add_argument('--abs_weight_fn', type=str, default='sqrt_inv', choices=['inv', 'sqrt_inv'],
        help="Function mapping (freq_abs, freq_presence) -> absence class weight, used when --class_weights_type=dataset_based.")

    parser.add_argument('--no_solarize', action='store_true', default=False)
    parser.add_argument('--n_glimpses', type=int, default=None, help='How long is each glimpse sequence.')
    parser.add_argument('--policy', type=str, default='cheat+jitter', help='which saccadic policy to use. humanlike or cheat+jitter or cheat')
    parser.add_argument('--train_shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8], help='Can either be a string of numerals 0123 or letters ABCD.')
    parser.add_argument('--map_shape_count', type=int, default=None, help='Override number of shape classes for map prediction head.')
    parser.add_argument('--test_shapes', nargs='*', type=list, default=[[0, 1, 2, 3, 5, 6, 7, 8], [4]]) 
    parser.add_argument('--min_num', type=int, default=1, help='minimum target numerosity')
    parser.add_argument('--max_num', type=int, default=5, help='maximum target numerosity')
    # parser.add_argument('--min_pass', type=int, default=0)
    # parser.add_argument('--max_pass', type=int, default=6) # related to symbolic model, which hasn't been updated 
    parser.add_argument('--transform', action='store_true', default=False, help='Whether to load augmented data with transposed and mirror images.')

    # Curriculum-mode dataset loading (mirrors the matching flags in datasets/dataset_generator.py --
    # these must be passed identically to how the dataset was generated, since loaders.py reconstructs
    # the dataset filename from them).
    parser.add_argument('--curriculum', type=str, default='', help='Curriculum preset name (e.g. 2to7). Empty = old non-curriculum loader path.')
    parser.add_argument('--curriculum_train_size', type=int, default=None,
        help='Size used when generating the curriculum-mode TRAIN file (must match --size passed to '
             'dataset_generator.py --dataset_split=train).')
    parser.add_argument('--curriculum_test_size', type=int, default=None,
        help='Size used when generating the 3 curriculum-mode TEST-condition files (must match --size passed to '
             'dataset_generator.py --dataset_split=test).')
    parser.add_argument('--curriculum_train_ratio', type=float, default=0.5)
    parser.add_argument('--curriculum_seed', type=int, default=None)
    parser.add_argument('--train_n_total_values', nargs='*', type=int, default=None)
    parser.add_argument('--test_n_total_values', nargs='*', type=int, default=None)
    parser.add_argument('--luminances', nargs='*', type=float, default=[0.1, 0.4, 0.7])
    parser.add_argument('--polar', action='store_true', default=False)
    parser.add_argument('--scaling', type=str, default='log')
    parser.add_argument('--no_glimpse', action='store_true', default=False)
    parser.add_argument('--truncate', action='store_true', default=False)
    parser.add_argument('--logscale', action='store_true', default=False)


    # Once loaded, how to prepare the data for training. What goes into the loaders?
    parser.add_argument('--place_code', action='store_true', default=False, help='Construct place code representation of glimpse coordinates.')
    parser.add_argument('--shape_input', type=str, default='symbolic', help='Which format to use for what pathway (symbolic, parametric, tetris, char, noise, logpolar)')
    parser.add_argument('--whole_image', action='store_true', default=False, help='whether to pass the whole image as input rather than glimpses')
    parser.add_argument('--sort', action='store_true', default=False, help='whether ventral stream was trained on sorted shape labels')
    parser.add_argument('--human_sim', action='store_true', default=False, help='Whether to run simulation of human experiment with fixed gaze.')
    parser.add_argument('--target_type', type=str, default='notA', help='If "all", target numerosity will include all items regardless of their identity.') 
    # parser.add_argument('--outer', action='store_true', default=False) # not currently implemented. Previously, whether to calculate the outer product of the two inputs to pass to the network
    # parser.add_argument('--rotate', action='store_true', default=False)  # not implemented. whether to apply a random rotation to the inputs. Especially relevant for sparse inputs (like place code) which can be difficult to learn from
    parser.add_argument('--misalign', action='store_true', default=False, help='Shuffle the glimpse positions relative to the glimpse contents.')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='rnn_classifier', help='rnn_classifier rnn_regression num_as_mapsum cnn')
    parser.add_argument('--train_on', type=str, default='both', help='Which input streams to include. xy, shape, or both')
    parser.add_argument('--h_size', type=int, default=25, help='how wide is the network in units. ')
    parser.add_argument('--pass_penult', action='store_true', default=False, help='whether to pass the penultimate layer of the ventral stream instead of the 2d prediction layer')
    parser.add_argument('--act', type=str, default=None, help='which activation function to use')
    parser.add_argument('--mult', action='store_true', default=False, help='Whether to train model with multiplicative interactions')
    parser.add_argument('--what_divisive_norm', type=str, default='none', choices=['none', 'fixed', 'learned'],
        help="Divisive normalization on the raw content ('what') stream before its embedding: 'none' disables it, "
             "'fixed' uses a constant semi-saturation constant, 'learned' makes it a trainable scalar.")
    parser.add_argument('--what_divisive_sigma', type=float, default=1.0, help='Semi-saturation constant for --what_divisive_norm.')
    parser.add_argument('--what_divisive_eps', type=float, default=1e-6, help='Numerical floor for --what_divisive_norm.')
    parser.add_argument('--rnn_type', type=str, default='vanilla', choices=['vanilla', 'gru', 'lstm'],
        help="Recurrent cell used by RNNClassifier2stream. 'vanilla' matches the original Elman-RNN behavior; "
             "'gru'/'lstm' use nn.GRUCell/nn.LSTMCell instead (see modules.py).")
    parser.add_argument('--use_layernorm', action='store_true', default=False,
        help='Apply LayerNorm to the hidden state each timestep (only affects rnn_type=gru or lstm).')
    parser.add_argument('--small_model_encoder', action='store_true', default=False,
        help="Replace pix_embedding/xy_embedding/joint_embedding with the small model's pre-RNN encoder shape "
             "(Linear(2,xy_preencode_size)+ReLU on xy, raw content, then Linear(.,joint_hidden_size)+ReLU / "
             "Linear(joint_hidden_size,hidden_size)+ReLU on the concatenation "
             "Parietal/ParietalInputEncoder). Only affects train_on='both'.")
    parser.add_argument('--xy_preencode_size', type=int, default=64,
        help="Width of the xy pre-encoder (--small_model_encoder only).")
    parser.add_argument('--joint_hidden_size', nargs='*', type=int, default=None,
        help="Layer widths for the model encoder's post-concatenation fusion stage "
             "(--small_model_encoder only; train_on='both'). If omitted, defaults to [256, h_size].")
    # parser.add_argument('--supervise_map', action='store_true', default=False, help='Whether to include a loss on the map prediction during training')

    # Training params
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--use_loss', type=str, default='both', help='num, map or both')
    parser.add_argument('--ventral', type=str, default=None)  
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--no_pretrain', action='store_true', default=False, help='if True, no saved weights will be loaded (add finetune to model type to make sure ventral steam is updated during training)')
    parser.add_argument('--opt', type=str, default='SGD', help='which optimiser to use. SGD, Adam, or AdamW')
    parser.add_argument('--detach', action='store_true', default=False, help='if True, number loss only backpropigates to the map layer. In this case, learning of most parameters in the network is governed by the map objective.')
    parser.add_argument('--learn_shape', action='store_true', default=False, help='for the parametric shape rep, whether to additional train to produce symbolic shape labels')
    parser.add_argument('--wd', type=float, default=0) # 1e-6
    parser.add_argument('--lr', type=float, default=None,
        help='Learning rate. If omitted, falls back to the hardcoded per-optimizer default in trainers.py '
             '(0.01 for SGD, 0.001 for Adam) -- unchanged from before this flag was wired up.')
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'flat'],
        help="'step': StepLR decay (x0.7 every n_epochs/20 epochs, reaching ~0.0008x the starting LR by "
             "the end regardless of how many epochs are requested). 'flat': no decay, constant LR throughout.")
    # parser.add_argument('--map_neg_w', type=float, default=0.1, help='Weight for absent shape channels in the map loss.')
    # parser.add_argument('--map_focal_gamma', type=float, default=2.0, help='Focal loss gamma for the per-shape map BCE.')

    # parser.add_argument('--n_iters', type=int, default=1, help='how many times the rnn should loop through sequence')
    # parser.add_argument('--small_weights', action='store_true', default=False)  # not implemented

    # Saving
    parser.add_argument('--rep', type=int, default=0, help='Identifier to distinguish between repetitions of the same experiment.')
    parser.add_argument('--save_act', action='store_true', default=False, help='whether to save network activations. recurrent layer if rnn or premap layer if cnn.')
    parser.add_argument('--if_exists', type=str, default='ask', help='What to do if results for this config already exist? "skip", "force" overwrite, or "ask" to increase rep counter.')
    parser.add_argument('--save_batch_confusion', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')


    # Hardware
    parser.add_argument('--no_cuda', action='store_true', default=False, help='if True, run the model on the cpu instead of gpu.')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')


    config = parser.parse_args()
    # config.n_classes = len(config.train_shapes)
    config.solarize = False if config.no_solarize else True
    #config.task_type = 'min' if config.challenge == 'min' else 'count'
    config.map_classes = None
    if config.model_type == 'rnn_regression':
        config.cross_entropy = False
    else:
        config.cross_entropy = True
    # Convert string input argument into a list of indices
    if config.train_shapes[0].isnumeric():
        config.shapestr = config.train_shapes.copy()
        config.testshapestr = config.test_shapes.copy()
        config.train_shapes = [int(i) for i in config.train_shapes]
        for j, test_set in enumerate(config.test_shapes):
            config.test_shapes[j] = [int(i) for i in test_set]
    elif config.train_shapes[0].isalpha():
        letter_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
                      'J':8, 'K':9, 'N':10, 'O':11, 'P':12, 'R':13, 'S':14,
                      'U':15, 'Z':16}
        config.shapestr = config.train_shapes.copy()
        config.testshapestr = config.test_shapes.copy()
        config.train_shapes = [letter_map[i] for i in config.train_shapes]
        for j, test_set in enumerate(config.test_shapes):
            config.test_shapes[j] = [letter_map[i] for i in test_set]
    if config.map_shape_count is None:
        config.map_shape_count = len(config.train_shapes)
        print(f'Setting map shape count to number of training shapes: {config.map_shape_count}')

    # Resolve curriculum-mode total-count value lists the same way dataset_generator.py does,
    # so loaders.py reconstructs the exact filenames the generator wrote.
    default_n_total_values = list(range(config.min_num, config.max_num + 1))
    config.train_n_total_values_list = (
        [int(v) for v in config.train_n_total_values]
        if config.train_n_total_values is not None and len(config.train_n_total_values) > 0
        else default_n_total_values
    )
    config.test_n_total_values_list = (
        [int(v) for v in config.test_n_total_values]
        if config.test_n_total_values is not None and len(config.test_n_total_values) > 0
        else default_n_total_values
    )
    config.curriculum_seed = config.seed if config.curriculum_seed is None else config.curriculum_seed

    if 'ventral' in config.model_type and config.no_pretrain:
        assert 'finetune' in config.model_type  # otherwise the params in the ventral module will never be trained!

    if config.joint_hidden_size is None:
        config.joint_hidden_size = [256, config.h_size]

    print(config)
    return config
