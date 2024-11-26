from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    freq = args.freq

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1#args.batch_size #1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.task_name == 'anomaly_detection':
        drop_last = False
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=args.percent,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=drop_last,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        # data_loader = DataLoader(
        #     data_set,
        #     batch_size=batch_size,
        #     shuffle=shuffle_flag,
        #     num_workers=args.num_workers,
        #     drop_last=drop_last)
        return data_set, data_loader
