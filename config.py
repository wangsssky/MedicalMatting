# configuration for the models
import yaml


class Config:
    def __init__(self, config_path):

        with open(config_path, encoding='utf-8') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        # ----------- parse yaml ---------------#
        self.DATA_PATH = yaml_dict['DATA_PATH']
        self.DATASET = yaml_dict['DATASET']

        if 'MASK_NUM' in yaml_dict:
            self.MASK_NUM = yaml_dict['MASK_NUM']
        else:
            if self.DATASET == 'brain-growth':
                self.MASK_NUM = 7
                self.INPUT_CHANNEL = 1
                self.INPUT_SIZE = 128
            elif self.DATASET == 'lidc':
                self.MASK_NUM = 4
                self.INPUT_CHANNEL = 1
                self.INPUT_SIZE = 128
            elif self.DATASET == 'isic':
                self.MASK_NUM = 3
                self.INPUT_CHANNEL = 3
                self.INPUT_SIZE = 256
            else:
                raise ValueError('unsupport dataset {}'.format(self.DATASET))
        print('MASK_NUM:', self.MASK_NUM)

        if 'LEVEL' in yaml_dict:
            self.LEVEL = yaml_dict['LEVEL']
            print('LEVEL:', self.LEVEL)
        else:
            self.LEVEL = None

        self.KFOLD = yaml_dict['KFOLD']
        self.RANDOM_SEED = yaml_dict['RANDOM_SEED']

        self.USE_MATTING = yaml_dict['USE_MATTING']
        if self.USE_MATTING:
            self.MODEL_NAME = 'ProbUnet_Matting'
        else:
            self.MODEL_NAME = 'ProbUnet'
        self.MODEL_DIR = yaml_dict['MODEL_DIR'] + self.MODEL_NAME
        self.UNCERTAINTY_MAP = yaml_dict['UNCERTAINTY_MAP']

        self.EPOCH_NUM = yaml_dict['EPOCH_NUM']
        self.RESUME_FROM = yaml_dict['RESUME_FROM']
        self.TRAIN_MATTING_START_FROM = yaml_dict['TRAIN_MATTING_START_FROM']

        self.TRAIN_BATCHSIZE = yaml_dict['TRAIN_BATCHSIZE']
        self.VAL_BATCHSIZE = yaml_dict['VAL_BATCHSIZE']
        self.TRAIN_TIME_AUG = yaml_dict['TRAIN_TIME_AUG']

        self.OPTIMIZER = yaml_dict['OPTIMIZER']
        self.WEIGHT_DECAY = yaml_dict['WEIGHT_DECAY']
        self.MOMENTUM = yaml_dict['MOMENTUM']
        self.LEARNING_RATE = float(yaml_dict['LEARNING_RATE'])
        self.WARM_LEN = yaml_dict['WARM_LEN']

        self.GEN_TYPE = yaml_dict['GEN_TYPE']
        self.NUM_FILTERS = yaml_dict['NUM_FILTERS']
        self.LATENT_DIM = yaml_dict['LATENT_DIM']
        self.SAMPLING_NUM = yaml_dict['SAMPLING_NUM']
        self.USE_BN = yaml_dict['USE_BN']
        self.POSTERIOR_TARGET = yaml_dict['POSTERIOR_TARGET']

        # self.REG_SCALE = float(yaml_dict['REG_SCALE'])
        self.KL_SCALE = float(yaml_dict['KL_SCALE'])
        self.RECONSTRUCTION_SCALE = yaml_dict['RECONSTRUCTION_SCALE']
        self.ALPHA_SCALE = yaml_dict['ALPHA_SCALE']
        self.ALPHA_GRADIENT_SCALE = yaml_dict['ALPHA_GRADIENT_SCALE']
        self.LOSS_STRATEGY = yaml_dict['LOSS_STRATEGY']

        self.PRT_LOSS = yaml_dict['PRT_LOSS']
        self.VISUALIZE = yaml_dict['VISUALIZE']


if __name__ == '__main__':
    cfg = Config(config_path='./params.yaml')
    print(cfg)




