import logging
import sys
sys.path.insert(0, "./")

class DividedDataset:
    def __init__(self, args):
        self.args = args
        self.train_data_local_dict = None
        self.test_data_local_dict = None
        self.dataset = None

    def load_data(self):
        """Load target data according to user's configuration."""
        logging.info("load_data. dataset_name = %s" % self.args.dataset_name)
        if self.args.dataset_name == "mnist":
            from data_preprocessing.MNIST.data_loader import load_partition_data_mnist
            data_loader = load_partition_data_mnist
        elif self.args.dataset_name == "fashion-mnist":
            from data_preprocessing.FashionMNIST.data_loader import load_partition_data_fashion_mnist
            data_loader = load_partition_data_fashion_mnist
        elif self.args.dataset_name == "cifar10":
            from data_preprocessing.CIFAR10.data_loader import load_partition_data_cifar10
            data_loader = load_partition_data_cifar10
        else:
            raise ValueError(f"dataset {self.args.dataset_name} have not been supported.")

        self.dataset = data_loader(self.args.data_dir, self.args.client_num_in_total,
                                   self.args.client_data_class, self.args.batch_size)
        
        self.train_num = self.dataset[0]
        self.test_num = self.dataset[1]
        self.train_data_local_dict = self.dataset[5]
        self.test_data_local_dict = self.dataset[6]
    
    @property
    def num_train(self):
        if self.train_num is None:
            raise ValueError("Data not loaded yet. Call `load_data` first.")
        return self.train_num
    
    @property
    def num_test(self):
        if self.test_num is None:
            raise ValueError("Data not loaded yet. Call `load_data` first.")
        return self.test_num

    @property
    def train_loader(self):
        if self.train_data_local_dict is None:
            raise ValueError("Data not loaded yet. Call `load_data` first.")
        return self.train_data_local_dict

    @property
    def test_loader(self):
        if self.test_data_local_dict is None:
            raise ValueError("Data not loaded yet. Call `load_data` first.")
        return self.test_data_local_dict


# Example usage:
# args = some_argument_parser()
# dataset = CustomDataset(args)
# dataset.load_data()
# train_data = dataset.train_loader
# test_data = dataset.test_loader
