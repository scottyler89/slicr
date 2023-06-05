import torch


class AnalysisResults:
    def __init__(self):
        self.obs_knn_adj_list_log = []
        self.corrected_obs_knn_dist_log = []
        self.knn_mask_log = []
        self.abs_beta_list_log = []
        self.total_beta_log = []
        self.adj = None
        self.dist = None
        self.mask = None

    def log_results(self, obs_knn_adj_list, corrected_obs_knn_dist, knn_mask, abs_beta_list, total_beta):
        self.obs_knn_adj_list_log.append(obs_knn_adj_list)
        self.corrected_obs_knn_dist_log.append(corrected_obs_knn_dist)
        self.knn_mask_log.append(knn_mask)
        self.abs_beta_list_log.append(abs_beta_list)
        self.total_beta_log.append(total_beta)
