from .dist import _init_dist_pytorch, get_dist_info, master_only, get_rank, collect_results_cpu
from .refcoco_refer import REFER
from .utils_refcoco import AverageMeter, Summary, intersectionAndUnionGPU