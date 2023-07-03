import torch


@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

class Metrics(object):
    """
    Define metrics for evaluation, metrics include:

        - MSE, masked MSE;

        - RMSE, masked RMSE;

        - REL, masked REL;

        - MAE, masked MAE;

        - Threshold, masked threshold.
    """
    def __init__(self, epsilon = 1e-8, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(Metrics, self).__init__()
        self.epsilon = epsilon
    
    def MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth

        Returns
        -------

        The MSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()
    
    def RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        RMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The RMSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2, dim = [1, 2])
        return torch.mean(torch.sqrt(sample_mse)).item()
    
    def MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted

        gt: tensor, required, the ground-truth

        Returns
        -------
        
        The MAE metric.
        """
        sample_mae = torch.mean(torch.abs(pred - gt))
        return sample_mae.item()

    # def WRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
    #     """
    #     WRMSE metric.

    #     Parameters
    #     ----------

    #     pred: tensor, required, the predicted;

    #     gt: tensor, required, the ground-truth;


    #     Returns
    #     -------

    #     The WRMSE metric.
    #     """
    #     return weighted_rmse_torch(pred, gt)

    def WRMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        WRMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The WRMSE metric.
        """

        return weighted_rmse_torch(pred, gt) * data_std

    # def WACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
    #     """
    #     WACC metric.

    #     Parameters
    #     ----------

    #     pred: tensor, required, the predicted;

    #     gt: tensor, required, the ground-truth;


    #     Returns
    #     -------

    #     The WACC metric.
    #     """
    #     return weighted_acc_torch(pred, gt)

    def WACC(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        WACC metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The WACC metric.
        """

        return weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)
    
    def APETM(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        #sample_mse = torch.mean(torch.abs(pred - gt)) + 0.04 * torch.mean((pred - gt) ** 2)
        sample_mse = torch.mean(torch.abs(pred - gt)) + 0.02 * torch.mean((pred - gt) ** 2)
        return sample_mse.item()

class MetricsRecorder(object):
    """
    Metrics Recorder.
    """
    def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        metrics_list: list of str, required, the metrics name list used in the metric calcuation.

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(MetricsRecorder, self).__init__()
        self.epsilon = epsilon
        self.metrics = Metrics(epsilon = epsilon)
        self.metrics_list = []
        for metric in metrics_list:
            try:
                metric_func = getattr(self.metrics, metric)
                self.metrics_list.append([metric, metric_func, {}])
            except Exception:
                raise NotImplementedError('Invalid metric type.')
    
    def evaluate_batch(self, data_dict):
        """
        Evaluate a batch of the samples.

        Parameters
        ----------

        data_dict: pred and gt


        Returns
        -------

        The metrics dict.
        """
        pred = data_dict['pred']            # (B, C, H, W)
        gt = data_dict['gt']
        data_mask = None
        clim_time_mean_daily = None
        data_std = None
        if "clim_mean" in data_dict:
            clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
            data_std = data_dict["std"]

        losses = {}
        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            loss = metric_func(pred, gt, data_mask, clim_time_mean_daily, data_std)
            if isinstance(loss, torch.Tensor):
                for i in range(len(loss)):
                    losses[metric_name+str(i)] = loss[i].item()
            else:
                losses[metric_name] = loss

        return losses
