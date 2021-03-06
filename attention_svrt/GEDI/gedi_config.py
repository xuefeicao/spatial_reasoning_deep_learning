from os.path import join as pjoin

class GEDIconfig(object): #at some point use pjoin throughout
    def __init__(self,which_dataset):

        self.which_dataset = which_dataset #gedi gfp masked_gfp or ratio

        #Parameters for extracting image patches from supplied TIFF files
        self.im_ext = '.png' #preserve floating point

        #Paths for creating tfrecords.
        self.GEDI_path = '/media/data/xuefei/svrt/'
        self.GEDI_path_1='/media/data_cifs/xuefei/'
        self.label_directories = [self.which_dataset + '_False',self.which_dataset + '_True'] 
        self.train_directory = self.GEDI_path + 'train/' + self.which_dataset + '/'
        self.validation_directory = self.GEDI_path + 'validation/' + self.which_dataset + '/'
        self.tfrecord_dir = self.GEDI_path + 'tfrecords/' + self.which_dataset + '/'
        self.tvt_flags = ['train','val'] #['train','val','test']

        #Data parameters for tfrecords
        self.train_proportion = 0.9 #validation with 10% of data
        self.num_threads = 2
        self.train_shards = 1#024 #number of training images per record
        self.validation_shards = 1#024 #number of training images per record
        self.train_batch = 32#number of training images per record
        self.validation_batch = 32
        self.normalize = True #Normalize GEDIs in uint8 to 0-1 float. Should be hardcoded...

        #Model training
        self.src_dir = '/home/xuefei/Project_spatial_reasoning/attention_svrt/'
        self.src_dir_1='/home/xuefei/Project_spatial_reasoning/vgg16_svrt/'
        self.epochs = 2 #Increase since we are augmenting
        self.train_checkpoint = self.GEDI_path + 'train_checkpoint/'
        self.train_summaries = self.GEDI_path + 'train_summaries/'
        self.vgg16_weight_path = pjoin(self.src_dir_1, 'pretrained_weights', 'vgg16.npy')
        self.model_image_size = [128,128,3]
        self.enc_size=256
        self.read_n=48
        self.T=3
        self.output_shape = 2 #how many categories for classification
        self.new_lr = 0.001
        self.data_augmentations = ['left_right','up_down','rotate'] #choose from: left_right, up_down, random_crop, random_brightness, random_contrast

        #Model testing
        self.results = pjoin(self.GEDI_path,'results/')
