class param(object):
    def __init__(self):
        # Path
        self.ROOT = 'Z:/2nd_paper'
        self.DATASET_PATH = f'{self.ROOT}/dataset/ND/PROPOSED_GAN_trainable_adj_Perceptual_HF'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/ViG_try1/ND/ckp'
        self.OUTPUT_LOSS = f'{self.ROOT}/backup/ViG_try1/ND/log'
        self.CKP_LOAD = False

        # Data
        self.DATA_STYPE = ['A', 'B']
        self.DATA_CLS = ['fake', 'live']
        self.SIZE = 224
        self.POOL_SIZE = 50

        # Train or Test
        self.EPOCH = 300
        self.LR = 1e-4
        self.B1 = 0.5
        self.B2 = 0.999
        self.BATCHSZ = 1
        self.PERCEPTUAL = True

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0