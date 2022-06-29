from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results', help='saves results here.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=100, help='how many test images to run')       
        self.parser.add_argument('--corrwise', action='store_true', help='if specified, use corrwise loss')

        self.isTrain = False
