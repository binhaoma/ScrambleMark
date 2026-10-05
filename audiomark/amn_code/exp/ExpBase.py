from pathlib import Path
import pickle
from datetime import datetime


class ExpBase:
    def __init__(self, out_dir, config, exp_status_fname, check_lst_fname="adv_lst.check"):
        """
        """
        self.out_dir = Path(out_dir)
        self.config = config

        # create folder if not exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # load experiment status if exists
        self.exp_status_path = self.out_dir.joinpath(exp_status_fname)
        self.status_dic = {}
        if self.exp_status_path.exists():
            with open(self.exp_status_path, 'rb') as handle:
                self.status_dic = pickle.load(handle)

        # load check list if exists
        self.check_lst_path = self.out_dir.joinpath(check_lst_fname)
        self.check_lst = []
        if self.check_lst_path.exists():
            with open(self.check_lst_path, 'rb') as handle:
                self.check_lst = pickle.load(handle)

    def save_status(self):
        # save experiment status
        with open(self.exp_status_path, 'wb') as handle:
            pickle.dump(self.status_dic, handle)

    def add_check(self, ae_path, target_phrase):
        """
        generated AEs should be added into check list so that we can check the correctness of AEs
        """
        self.check_lst.append([ae_path, target_phrase])

        with open(self.check_lst_path, 'wb') as handle:
            pickle.dump(self.check_lst, handle)

    def save_flag(self, flag_path, flag=None):
        if flag is None:
            flag = datetime.now()

        with open(flag_path, 'wb') as handle:
            pickle.dump(flag, handle)

    def load_flag(self, flag_path):
        if not flag_path.exists():
            return None

        with open(flag_path, 'rb') as handle:
            flag = pickle.load(handle)
        return flag

    def run(self):
        """
        run the experiments
        """
        raise NotImplementedError
