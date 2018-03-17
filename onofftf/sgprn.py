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
                   u_wm_list,u_ws_sqrt_list,wkern_list,Zw_list,
                   u_gm_list,u_gs_sqrt_list,gkern_list,Zg_list):
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

    kl_g = [[GaussKL(u_gm_list[j][i].get_tfv(),
                    u_gs_sqrt_list[j][i].get_tfv(),
                    gkern_list[j][i].K(Zg_list[j][i].get_tfv()))
            for i in range(num_output)]
            for j in range(num_latent)]

    kl = tf.add_n(kl_f) + tf.reduce_sum(tf.add_n(kl_w)) + tf.reduce_sum(tf.add_n(kl_g))

    return kl


# Build predictive posterior
def build_predict(Xnew,u_fm_list,u_fs_sqrt_list,fkern_list,Zf_list,
                       u_wm_list,u_ws_sqrt_list,wkern_list,Zw_list,
                       u_gm_list,u_gs_sqrt_list,gkern_list,Zg_list):
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

    # get conditional for g
    gmean_list = [[[] for i in range(num_output)]
                      for j in range(num_latent)]

    gvar_list  = [[[] for i in range(num_output)]
                      for j in range(num_latent)]
    for i in range(num_output):
        for j in range(num_latent):
            gmean_list[j][i], gvar_list[j][i] = GPConditional(Xnew,
                                                              Zg_list[j][i].get_tfv(),
                                                              gkern_list[j][i],
                                                              u_gm_list[j][i].get_tfv(),
                                                              full_cov=False,
                                                              q_sqrt=u_gs_sqrt_list[j][i].get_tfv(),
                                                              whiten=False)
    gmean = tf.concat(gmean_list,axis=2)
    gvar = tf.concat(gvar_list,axis=2)

    # Get probit expectation for \g
    pgmean, pgvar = probit_expectations(gmean,gvar)

    # Augment W with sparity function \g
    wgmean = tf.multiply(wmean,pgmean)

    # compute augmented f : wf : (w \circ pgmean)^T f
    wfmean = tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(fmean,axis=0),wgmean),axis=2))

    return wfmean, fmean, fvar, wmean, wvar, pgmean, pgvar, wgmean


# variational expectations of likelihood
def variational_expectations(Y, wfmean, fmean, fvar, wmean, wvar, pgmean, pgvar, noisevar_list):

    N = tf.cast(tf.shape(Y)[0], float_type)
    noisevar = tf.stack([var.get_tfv() for var in noisevar_list],axis=0)

    t1 = tf.reduce_sum(tf.square(tf.subtract(Y,wfmean)),axis=0)

    # Tr [fmean^2 {(pgmean^2 + pgvar)  \circ wvar }]
    t2 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(tf.square(fmean),axis=0),tf.multiply(tf.add(tf.square(pgmean),pgvar),wvar)),axis=2)),axis=0)
    # Tr [fmean^2 { pgvar \circ wvar }]
    t3 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(tf.square(fmean),axis=0),tf.multiply(pgvar,tf.square(wmean))),axis=2)),axis=0)
    # Tr [fvar {(pgmean^2 + pgvar) \circ wvar }]
    t4 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(fvar,axis=0),tf.multiply(tf.add(tf.square(pgmean),pgvar),tf.square(wmean))),axis=2)),axis=0)
    # Tr [fvar {(pgmean^2 + pgvar) \circ wvar }]
    t5 = tf.reduce_sum(tf.transpose(tf.reduce_sum(tf.multiply(tf.expand_dims(fvar,axis=0),tf.multiply(tf.add(tf.square(pgmean),pgvar),wvar)),axis=2)),axis=0)

    tsum = tf.divide(t1 + t2 + t3 + t4 + t5,noisevar)

    return - 0.5 * N * tf.reduce_sum(tf.log(2 * np.pi * noisevar)) \
           - tf.reduce_sum(0.5 * tsum)


def probit_expectations(gmean, gvar):
    def normcdf(x):
        return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1. - 2.e-3) + 1.e-3

    def owent(h, a):
        h = tf.abs(h)
        term1 = tf.atan(a) / (2 * np.pi)
        term2 = tf.exp((-1 / 2) * (tf.multiply(tf.square(h), (tf.square(a) + 1))))
        return tf.multiply(term1, term2)

    z = gmean / tf.sqrt(1. + gvar)
    a = 1 / tf.sqrt(1. + (2 * gvar))

    cdfz = normcdf(z)
    tz = owent(z, a)

    ephig = cdfz
    evarphig = (cdfz - 2. * tz - tf.square(cdfz))

    # clip negative values from variance terms to zero
    evarphig = (evarphig + tf.abs(evarphig)) / 2.

    return ephig, evarphig

# gamma piror for the lengthscale
def ell_gamma_pior(ell,prior_alpha,prior_beta):
    return tf.cast(prior_alpha * tf.log(prior_beta),dtype=float_type) + \
           tf.cast((prior_alpha - 1.0),dtype=float_type) * tf.log(ell) - \
           prior_beta * ell - tf.cast(tf.lgamma(prior_alpha),dtype=float_type)


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


def plot_sgprn_fit(Xtrain,Ytrain,xplot,yplot,pred_plt_wfmean,pred_plt_fmean,pred_plt_wgmean,
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
            cspace = np.linspace(-pred_plt_wgmean[wf,:,lf].max(),pred_plt_wgmean[wf,:,lf].max(),10)
            norm = mpl.colors.Normalize(vmin=-pred_plt_wgmean[wf,:,lf].max(),vmax=pred_plt_wgmean[wf,:,lf].max())
            ax   = fig.add_subplot(gs01[wf,lf],adjustable='box-forced')
            cax  = ax.contourf(xplot, yplot,pred_plt_wgmean[wf,:,lf].reshape(xplot.shape),
                               cmap='bwr',alpha=0.7,norm=norm,
                              levels = cspace)#,cmap='Blues'
            ax.contour(xplot,yplot,pred_plt_wgmean[wf,:,lf].reshape(xplot.shape),levels=[cspace[4],cspace[5]],linewidths=1,colors='black',linestyles='solid')
            cbar = fig.colorbar(cax,extend='max',fraction=0.046,pad=0.01)
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
    plt.figtext(0.38,0.9, "(b) Sparse weight functions", ha="center", va="center", fontsize=18)
    plt.figtext(0.68,0.9, "(c) Predictive \n  posteriors", ha="center", va="center", fontsize=18)
    plt.figtext(0.86,0.9, "(d) Actual", ha="center", va="center", fontsize=18)
    plt.savefig("plots/sgprn_fit.png")
    plt.show()


def plot_sgprn_augmentation(xplot,yplot,pred_plt_wmean,pred_plt_pgmean,ind_plot_g,ind_plot_w):
    num_latent = 2
    num_output = 3

    mpl.rcParams['figure.figsize'] = (10,7)
    tick_locator = ticker.MaxNLocator(nbins=4)
    mpl.rcParams.update({'font.size': 16})

    x1min = 0.5;x1max = 6.0;x2min =0.5;x2max = 6.0;
    fig = plt.figure()

    # plot weight functions
    gs01 = gridspec.GridSpec(num_output,num_latent)
    for lf in np.arange(num_latent):
        for wf in np.arange(num_output):
            norm = mpl.colors.Normalize(vmin=0,vmax=1)
            ax   = fig.add_subplot(gs01[wf,lf],adjustable='box-forced')
            cax  = ax.contourf(xplot, yplot,pred_plt_pgmean[wf,:,lf].reshape(xplot.shape),cmap='Greys',alpha=0.7,norm=norm)#,cmap=''
            cbar = fig.colorbar(cax,extend='max',fraction=0.046, pad=0.04,ticks=[0,0.5,1])
            ax.scatter(ind_plot_g[lf][wf][:,0],ind_plot_g[lf][wf][:,1],s=10)
            ax.set_xlim(x1min,x1max)
            ax.set_ylim(x2min,x2max)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    #         cbar.locator = tick_locator
    #         cbar.update_ticks()
            ax.set_title(str([wf+1,lf+1]))


    gs01.tight_layout(fig, rect=[0, 0, 0.5,0.88],h_pad=0.5)

    # plot weight functions
    gs01 = gridspec.GridSpec(num_output,num_latent)
    for lf in np.arange(num_latent):
        for wf in np.arange(num_output):
            norm = mpl.colors.Normalize(vmin=-pred_plt_wmean[wf,:,lf].max(),vmax=pred_plt_wmean[wf,:,lf].max())
            ax   = fig.add_subplot(gs01[wf,lf],adjustable='box-forced')
            cax  = ax.contourf(xplot, yplot,pred_plt_wmean[wf,:,lf].reshape(xplot.shape),cmap='YlOrBr',alpha=0.7,norm = norm)#,cmap=''
            cbar = fig.colorbar(cax,extend='max',fraction=0.046, pad=0.04)
            ax.scatter(ind_plot_w[lf][wf][:,0],ind_plot_w[lf][wf][:,1],s=10)
            ax.set_xlim(x1min,x1max)
            ax.set_ylim(x2min,x2max)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            cbar.locator = tick_locator
            cbar.update_ticks()
            ax.set_title(str([wf+1,lf+1]))

    gs01.tight_layout(fig, rect=[0.5, 0,1,0.88],h_pad=0.5)

    plt.figtext(0.24,0.9, "(a) Probit support functions ", ha="center", va="center", fontsize=18)
    plt.figtext(0.73,0.9, "(b) Latent weight functions", ha="center", va="center", fontsize=18)
    plt.savefig("plots/sgprn_aug.png")
    plt.show()
