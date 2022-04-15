import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='pascal_voc_0712', type=str)
    parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset',
                        default='clipart', type=str)
    parser.add_argument('--dataset_a', dest='dataset_a',
                        help='target training dataset',
                        default='clipart', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101 res50',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=5, type=float)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--adaptation', dest='adaptation',
                        help='the kind of adaptation, e.g. so or ins_style',
                        default='adap', type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')

    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        action='store_true')
    parser.add_argument('--ic', dest='ic',
                        help='whether use context vector for global level with pixelD',
                        action='store_true')
    parser.add_argument('--cr', dest='cr',
                        help='whether use consistency regularization between image and instance adaptation',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-3, type=float)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)
    parser.add_argument('--eta_uplmt', dest='eta_uplmt',
                        help='the upper limit factor of eta value',
                        default=1, type=float)
    parser.add_argument('--eta_style', dest='eta_style',
                        help='the upper limit factor of eta value',
                        default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--tfb_path', dest='tfb_path',
                        help='path to save tfb logs',
                        type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--proposal_dir', dest='proposal_dir',
                        help='directory to save proposals', default="./data/feats",
                        type=str)

    # style transfer
    parser.add_argument('--transform_method', dest='transform_method',
                        help='the transform method for making target size same as source', default="scale",
                        type=str)
    parser.add_argument('--style_lambda', dest='style_lambda',
                        help='the banlance factor for style loss', default=1e7,
                        type=float)

    # vrd
    parser.add_argument('--num_classes', dest='num_classes',
                        help='the number of object class in vrd', default=15,
                        type=int)
    parser.add_argument('--num_relations', dest='num_relations',
                        help='the number of relation class in vrd', default=62,
                        type=int)
    parser.add_argument('--source_so_prior_path', dest='source_so_prior_path',
                        help='path of source so prior konwledge',
                        default="./data/MVidVRD/source_so_prior.pkl")
    parser.add_argument('--source_gt_rels_path', dest='source_gt_rels_path',
                        help='path of source gt relation labels',
                        default="./data/MVidVRD/source_gt_rels.pkl") 
    # parser.add_argument('--target_so_prior_path', dest='target_so_prior_path',
    #                     help='path of target so prior konwledge',
    #                     default="./data/target_so_prior.pkl")
    parser.add_argument('--target_gt_rels_path', dest='target_gt_rels_path',
                        help='path of target gt relation labels',
                        default="./data/MVidVRD/target_gt_rels.pkl") 
    parser.add_argument('--use_obj_visual', dest='use_obj_visual',
                        help='use the object viusal features for vrd or not',
                        default=True, type=bool) 
    parser.add_argument('--use_semantic', dest='use_semantic',
                        help='use the semantic information between subject and object for vrd or not',
                        default=True, type=bool)
    parser.add_argument('--spatial_type', dest='spatial_type',
                        help='choose the method encoding spatial feature, 1-relative, 2-conv mask',
                        default=2, type=bool)   
    parser.add_argument('--vrd_task', dest='vrd_task',
                        help='the task type for vrd',
                        default="rel_det", type=str)
    parser.add_argument('--vrd_lr', dest='vrd_lr',
                        help='starting learning rate of vrd net',
                        default=1e-5, type=float)                                                                                                                     
    parser.add_argument('--glove_path', dest='glove_path',
                        help='directory to save models', default="./models/glove.6B.300d.txt",
                        type=str)
    parser.add_argument('--predicate_file', dest='predicate_file',
                        help='file to save predication', default="./data/MVidVRD/predicates.json",
                        type=str)
    parser.add_argument('--save_feat_path', dest='save_feat_path',
                        help='save_feat_path', default='./frame_feat',
                        type=str)
    parser.add_argument('--save_videofeat_path', dest='save_videofeat_path',
                        help='save_videofeat_path', default='./video_feat',
                        type=str)
    parser.add_argument('--emb_dim', dest='emb_dim',
                        help='the dim of embedding space', default=300,
                        type=int)

    parser.add_argument('--semi', dest='semi',
                        help='visualization mode',
                        action='store_true')
                        
    args = parser.parse_args()
    return args

def set_dataset_args(args, test=False):
    if not test:
        if args.dataset == "MVRD":
            args.imdb_name = "MVRD_trainval"
            args.imdbval_name = "MVRD_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30']
        if args.dataset_t == "MVidVRD":
            args.imdb_name_target = "MVidVRD_val"
            args.imdbval_name_target = "MVidVRD_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30'] 
    else:
        if args.dataset == "MVidVRD":
            args.imdb_name = "MVidVRD_val"
            args.imdbval_name = "MVidVRD_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args
