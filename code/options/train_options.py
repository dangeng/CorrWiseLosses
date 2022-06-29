from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=20, help='# of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

        # losses
        self.parser.add_argument('--perceptual', action='store_true', help='if specified, do use perceptual loss')        
        self.parser.add_argument('--l1', action='store_true', help='if specified, do use perceptual loss')        
        self.parser.add_argument('--mse', action='store_true', help='if specified, do use perceptual loss')

        # corrwise options
        self.parser.add_argument('--corrwise', action='store_true', help='if specified, use corrwise loss')        
        self.parser.add_argument('--padding_mode', type=str, default='reflection', help='type of padding to use when warping [zeros, border, reflection]')
        self.parser.add_argument('--flow_method', type=str, default='RAFT', help='flow method to use when warping [RAFT, RAFT-KITTI]')
        self.parser.add_argument('--epsilon', type=float, default=0.1, help='scale the flow by 1 - \epsilon ')
        self.parser.add_argument('--no_detach', action='store_true', help='if specified, do *not* detach')        

        # Regularization
        self.parser.add_argument('--reg_flow_mag', type=float, default=0, help="if non-zero, regularize by magnitude of flow, multiplied by this float's scale factor")
        self.parser.add_argument('--reg_flow_grad_mag', type=float, default=0, help="if non-zero, regularize by magnitude of gradient flow, multiplied by this float's scale factor")
        self.parser.add_argument('--flow_reg_norm', type=int, default=1, help='int that specifies the norm to calculate flow magnitude for regularization')
        self.parser.add_argument('--eas', type=int, default=0, help='The order of edge-aware smoothness [0, 1, 2]')
        self.parser.add_argument('--edge_penalty', type=float, default=0.0, help='penalty for out of bounds flow predictions')


        self.isTrain = True
