import argparse
from lib.utils import yaml2config
from networks import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        help="mode: [rand] [style] [interp] [text]",
    )

    # --- PHẦN THÊM MỚI ---
    parser.add_argument(
        "--text",
        nargs="?",
        type=str,
        default="htphong", # Giá trị mặc định nếu không nhập
        help="Input text content for generation",
    )
    # ---------------------

    args = parser.parse_args()
    cfg = yaml2config(args.config)

    model = get_model(cfg.model)(cfg, args.config)
    
    # Load model (lưu ý: đảm bảo bạn đã sửa hàm load trong model.py để tránh lỗi key)
    model.load(args.ckpt, cfg.device) 

    # Truyền args.text vào các hàm eval
    if args.mode == 'style':
        model.eval_style(text=args.text)
        
    elif args.mode == 'rand':
        model.eval_rand(text=args.text)
        
    elif args.mode == 'interp':
        model.eval_interp(text=args.text)
        
    elif args.mode == 'text':
        model.eval_text(text=args.text)
        
    else:
        print('Unsupported mode: {} | Use: [rand] [style] [interp] [text]'.format(args.mode))