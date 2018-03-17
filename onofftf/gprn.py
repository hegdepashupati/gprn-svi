import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import matplotlib.colors as colors


from .main import Param, GaussKL, KernSE, GPConditional

float_type = tf.float64
jitter_level = 1e-5

# KL divergence between all inducing priors and posteriors
def build_prior_kl(u_fm_list,u_fs_sqrt_list,fkern_list,Zf_list,
                   u_wm_list,u_ws_sqrt_list,wkern_list,Zw_list):
    num_output = len(Zw_list[0])
    num_latent = len(Zw_list)

    kl_f = [GaussKL(u_fm_list[i].get_tfv(),
                    u_fs_sqrt_list[i].get_tfv(),
                    fkern_list[i].K(Zf_list[i].get_tfv()))
            for i in range(num_latent)]

    kl_w = [[GaussKL(u_wm_list[j][i].get_tfv(),
                    u_ws_sqrt_list[j][i].get_tfv(),
                    wkern_list[j][i].K(Zw_list[j][i].get_tfv()))
            for i in range(num_output)]
            for j in range(num_latent)]

    kl = tf.add_n(kl_f) + tf.reduce_sum(tf.add_n(kl_w))

    return kl


# Build predictive posterior
def build_predict(Xnew,u_fm_list,u_fs_sqrt_list,fkern_list,Zf_list,
                       u_wm_list,u_ws_sqrt_list,wkern_list,Zw_list):
    num_output = len(Zw_list[0])
    num_latent = len(Zw_list)

    # Get conditionals for f
    fmean_list = [[] for i in range(num_latent)]
    fvar_list  = [[] for i in range(num_latent)]
    for i in range(num_latent):
        fmean_list[i], fvar_list[i] = GPConditional(Xnew,
                                                    Zf_list[i].get_tfv(),
                                                    fkern_list[i],
                                                    u_fm_list[i].get_tfv(),
                                                    full_cov=False,
                                                    q_sqrt=u_fs_sqrt_list[i].get_tfv(),
                                                    whiten=False)
    fmean = tf.concat(fmean_list,axis=1)
    fvar = tf.concat(fvar_list,axis=1)

    # get conditional for w
    wmean_list = [[[] for i in range(num_output)]
                      for j in range(num_latent)]

    wvar_list  = [[[] for i in range(num_output)]
                      for j in range(num_latent)]
    for i in range(num_output):
        for j in range(num_latent):
            wmean_list[j][i], wvar_list[j][i] = GPConditional(Xnew,
                                                              Zw_list[j][i].get_tfv(),
                                                              wkern_list[j][i],
                                                              u_wm_list[j][i].get_tfv(),
                                                              full_cov=False,
                                                              q_sqrt=u_ws_sqrt_list[j][i].get_tfv(),
                                                              whiten=False)
    wmean = tf.concat(wmean_list,axis=2)
    wvar = tf.concat(wvar_list,axis=2)

    # compute augmented f : wf : w^T f
    wfmean = tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(fmean,axis=0),wmean),axis=2))

    return wfmean, fmean, fvar, wmean, wvar


# variational expectations of likelihood
def variational_expectations(Y, wfmean, fmean, fvar, wmean, wvar, noisevar_list):
    N = tf.cast(tf.shape(Y)[0], float_type)
    noisevar = tf.stack([var.get_tfv() for var in noisevar_list],axis=0)

    t1 = tf.reduce_sum(tf.square(tf.subtract(Y,wfmean)),axis=0)
    t2 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(fvar,axis=0),tf.square(wmean)),axis=2)),axis=0)
    t3 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(tf.square(fmean),axis=0),wvar),axis=2)),axis=0)
    t4 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(fvar,axis=0),wvar),axis=2)),axis=0)

    tsum = tf.divide(t1 + t2 + t3 + t4,noisevar)

    return - 0.5 * N * tf.reduce_sum(tf.log(2 * np.pi * noisevar)) \
           - tf.reduce_sum(0.5 * tsum)


def generate_train_op(cost_op):
    all_var_list = tf.trainable_variables()
    all_lr_list = [var._learning_rate for var in all_var_list]

    train_opt_group = []

    for group_learning_rate in set(all_lr_list):
        _ind_bool = np.where(np.isin(np.array(all_lr_list),group_learning_rate))[0]
        group_var_list = [all_var_list[ind] for ind in _ind_bool]
        group_tf_optimizer = tf.train.AdamOptimizer(learning_rate = group_learning_rate)
        group_grad_list = tf.gradients(cost_op,group_var_list)
        group_grads_and_vars = list(zip(group_grad_list,group_var_list))


        group_train_op = group_tf_optimizer.apply_gradients(group_grads_and_vars)

        # Summarize all gradients
        for grad, var in group_grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', grad)

        train_opt_group.append({'names':[var.name for var in group_var_list],
                                'vars':group_var_list,
                                'learning_rate':group_learning_rate,
                                'grads':group_grad_list,
                                'train_op':group_train_op})

    train_op = tf.group(*[group['train_op'] for group in train_opt_group])
    return train_op


def plot_gprn_fit(Xtrain,Ytrain,xplot,yplot,pred_plt_wfmean,pred_plt_fmean,pred_plt_wmean,
             ind_plot_f,ind_plot_w):
    num_latent = pred_plt_fmean.shape[1]
    num_output = pred_plt_wfmean.shape[1]

    tick_locator = ticker.MaxNLocator(nbins=4)
    mpl.rcParams['figure.figsize'] = (12,7)
    mpl.rcParams.update({'font.size': 14})

    x1min = 0.5;x1max = 6.0;x2min =0.5;x2max = 6.0;

    fig = plt.figure()

    # plot latent functions
    gs00 = gridspec.GridSpec(num_latent, 1)
    for lf in np.arange(num_latent):
        ax   = fig.add_subplot(gs00[lf,0], aspect='equal')
        cax  = ax.contourf(xplot, yplot, pred_plt_fmean[:,lf].reshape(xplot.shape),cmap='RdYlBu',alpha=0.7)#cmap='RdYlBu'
        cbar = fig.colorbar(cax,extend='max',fraction=0.046,pad=0.01)
        ax.scatter(ind_plot_f[lf][:,0],ind_plot_f[lf][:,1],s=10)
        ax.set_xlim(x1min,x1max)
        ax.set_ylim(x2min,x2max)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        cbar.locator = tick_locator
        cbar.update_ticks()
        ax.set_title(str([lf+1]))
    gs00.tight_layout(fig, rect=[0.02, 0, 0.2, 0.88])

    # plot weight functions
    gs01 = gridspec.GridSpec(num_output,num_latent)
    for lf in np.arange(num_latent):
        for wf in np.arange(num_output):
            ax   = fig.add_subplot(gs01[wf,lf],adjustable='box-forced')
            cax  = ax.contourf(xplot, yplot,pred_plt_wmean[wf,:,lf].reshape(xplot.shape),alpha=0.7,cmap='bwr')
            cbar = fig.colorbar(cax,extend='max',fraction=0.046, pad=0.04)
            ax.scatter(ind_plot_w[lf][wf][:,0],ind_plot_w[lf][wf][:,1],s=10)
            ax.set_xlim(x1min,x1max)
            ax.set_ylim(x2min,x2max)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            cbar.locator = tick_locator
            cbar.update_ticks()
            ax.set_title(str([wf+1,lf+1]))
    gs01.tight_layout(fig, rect=[0.2, 0, 0.61,0.88],h_pad=0.5)

    vlims = [3,40,200]
    mlist = ['Cd','Ni','Zn']
    # plot output functions
    gs02 = gridspec.GridSpec(num_output, 1)
    for wf in np.arange(num_output):
    #     norm = mpl.colors.Normalize(vmin=0,vmax=vlims[wf])
        bounds = np.linspace(0, vlims[wf], 20)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        ax   = fig.add_subplot(gs02[wf,0], aspect='equal',adjustable='box-forced')
        cax  = ax.contourf(xplot, yplot,np.minimum(pred_plt_wfmean[:,wf],200).reshape(xplot.shape),alpha=0.7,norm=norm)
        cbar = fig.colorbar(cax,extend='max',fraction=0.046,pad=0.01)
        cbar.set_clim(0.,vlims[wf])
        ax.set_xlim(x1min,x1max)
        ax.set_ylim(x2min,x2max)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        cbar.locator = tick_locator
        cbar.update_ticks()
        ax.set_title(mlist[wf])


    gs02.tight_layout(fig, rect=[0.51, 0, 0.78, 0.88],h_pad=0.5)

    # plot data functions
    gs03 = gridspec.GridSpec(num_output, 1)
    for wf in np.arange(num_output):
        ax   = fig.add_subplot(gs03[wf,0], aspect='equal',adjustable='box-forced')
        cax  = ax.scatter(Xtrain[:,0],Xtrain[:,1],c= Ytrain[:,wf],alpha=0.7,s=10)
        cbar = fig.colorbar(cax,extend='max',fraction=0.046,pad=0.01)
        ax.set_xlim(x1min,x1max)
        ax.set_ylim(x2min,x2max)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        cbar.locator = tick_locator
        cbar.update_ticks()
        ax.set_title(mlist[wf])

    gs03.tight_layout(fig, rect=[0.78, 0, 0.95, 0.88],h_pad=0.01,w_pad=0.01)

    plt.figtext(0.1,0.9, "(a) Latent \n functions", ha="center", va="center", fontsize=18)
    plt.figtext(0.38,0.9, "(b) Mixing matrix functions", ha="center", va="center", fontsize=18)
    plt.figtext(0.68,0.9, "(c) Predictive \n  posteriors", ha="center", va="center", fontsize=18)
    plt.figtext(0.86,0.9, "(d) Actual", ha="center", va="center", fontsize=18)
    plt.savefig("plots/gprn_fit.png")
    plt.show()
