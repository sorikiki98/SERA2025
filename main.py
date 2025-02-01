import torchvision
import argparse
from torch.utils.data import DataLoader
from data.fashion200k import Fashion200kAttribute
from models import create_models

def load_dataset(opt):
    train_set = Fashion200kAttribute(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    )

    test_set = Fashion200kAttribute(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    )

    # todo: Fashion200kAttribute(parent-child), Fashion200kAttributeItem(parent-child), Fashion200kAttributeItem(sibling)

    print('train_set size:', len(train_set))
    print('test_set size:', len(test_set))

    return train_set, test_set


def main(cfg: RunConfig):
    opt = parse_opt()
    print('Arguments:')
    for k in opt.__dict__.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    train_set, test_set = load_dataset(opt)
    train_dataloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=opt.shuffle, drop_last=True,
                                  num_workers=opt.num_workers, pin_memory=True)

    # init optimizer and scheduler
    # init trainer

    # models = create_models(opt.__dict__)


if __name__ == '__main__':
    main()
