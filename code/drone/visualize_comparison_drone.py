import torch
import numpy as np
import pandas as pd
from model import PenguiNet
from ConvBlock import ConvBlock
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.animation as animation


def VizDroneBEV(frames, labels, bpred, opred, name, isGray=True):
    fig = plt.figure(888, figsize=(15, 5))
    h = 9
    w = 16

    ax0 = plt.subplot2grid((h, w), (0, 0), colspan=7, rowspan=2)
    ax0.set_ylim([0, 9])
    annotation = ax0.annotate("poop", xy=(0, 5.5), size=10)
    annotation.set_animated(True)
    ax0.axis('off')

    ax1 = plt.subplot2grid((h, w), (2, 0), colspan=8, rowspan=7)
    ax1.set_title('Relative Pose (x,y)')
    ax1.yaxis.set_ticks([0, 1.5, 3])  # set y-ticks
    ax1.xaxis.set_ticks([-3.0, -1.5, 0, 1.5, 3.0])  # set y-ticks
    ax1.xaxis.tick_top()  # and move the X-Axis
    ax1.yaxis.tick_left()  # remove right y-Ticks
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.invert_xaxis()
    trianglex = [2, 0, -2, 2]
    triangley = [3, 0, 3, 3]
    collection = plt.fill(trianglex, triangley, facecolor='lightskyblue')

    plot1gt, = plt.plot([], [], color='green', label='GT',
                        linestyle='None', marker='o', markersize=10)
    plot1bpr, = plt.plot([], [], color='blue', label='Baseline',
                         linestyle='None', marker='o', markersize=10)
    plot1opr, = plt.plot([], [], color='red', label='Ours',
                         linestyle='None', marker='o', markersize=10)
    arr1gt = ax1.arrow([], [], np.cos([]), np.sin(
        []), head_width=0.1, head_length=0.1, color='green', animated=True)
    arr1pr = ax1.arrow([], [], np.cos([]), np.sin(
        []), head_width=0.1, head_length=0.1, color='blue', animated=True)
    plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

    ax2 = plt.subplot2grid((h, w), (2, 8), rowspan=7)
    ax2.set_title('Relative z', pad=20)
    ax2.yaxis.tick_right()
    ax2.set_ylim([-1, 1])
    ax2.set_xticklabels([])
    ax2.yaxis.set_ticks([-1, 0, 1])  # set y-ticks
    ax2.xaxis.set_ticks_position('none')
    scatter2gthead, = plt.plot(
        [], [], color='green', linestyle='None', marker='o', markersize=10)
    scatter2bpredhead, = plt.plot(
        [], [], color='blue', linestyle='None', marker='o', markersize=10)
    scatter2opredhead, = plt.plot(
        [], [], color='red', linestyle='None', marker='o', markersize=10)

    ax3 = plt.subplot2grid((h, w), (2, 9), rowspan=7, colspan=7)
    ax3.axis('off')
    frame = frames[0].astype(np.uint8)
    if isGray == True:
        imgplot = plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
    else:
        frame = frame.transpose(1, 2, 0)
        imgplot = plt.imshow(frame)

    ax4 = plt.subplot2grid((h, w), (0, 9), colspan=7)
    ax4.set_xlim([0, 8])
    annotation2 = ax4.annotate("poop", xy=(3, 0.1), size=14, weight='bold')
    annotation2.set_animated(True)
    ax4.axis('off')

    plt.subplots_adjust(wspace=1.5)

    img = mpimg.imread('./drone/minidrone.jpg')
    newax = fig.add_axes([0.26, 0.0, 0.1, 0.1], anchor='S')
    newax.imshow(img)
    newax.axis('off')

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='FullMetalNicky'))

    def animate(id):
        x_gt, y_gt, z_gt, phi_gt = labels[id][0], labels[id][1], labels[id][2], labels[id][3]
        x_bpred, y_bpred, z_bpred, phi_bpred = bpred[id][0], bpred[id][1], bpred[id][2], bpred[id][3]
        x_opred, y_opred, z_opred, phi_opred = opred[id][0], opred[id][1], opred[id][2], opred[id][3]

        str1 = "x_gt={:05.3f}, y_gt={:05.3f}, z_gt={:05.3f}, phi_gt={:05.3f}\n".format(
            x_gt, y_gt, z_gt, phi_gt)
        str1 = str1 + "x_bs={:05.3f}, y_bs={:05.3f}, z_bs={:05.3f}, phi_bs={:05.3f}\n".format(
            x_bpred, y_bpred, z_bpred, phi_bpred)
        str1 = str1 + "x_sc={:05.3f}, y_sc={:05.3f}, z_sc={:05.3f}, phi_sc={:05.3f}".format(
            x_opred, y_opred, z_opred, phi_opred)

        phi_gt = - phi_gt - np.pi / 2
        phi_bpred = -phi_bpred - np.pi / 2
        phi_opred = -phi_opred - np.pi / 2

        annotation.set_text(str1)

        plot1gt.set_data(np.array([y_gt, x_gt]))
        plot1bpr.set_data(np.array([y_bpred, x_bpred]))
        plot1opr.set_data(np.array([y_opred, x_opred]))

        if(len(ax1.patches) > 1):
            ax1.patches.pop()
            ax1.patches.pop()
            ax1.patches.pop()

        patch1 = patches.FancyArrow(y_gt, x_gt, 0.5 * np.cos(phi_gt), 0.5 * np.sin(
            phi_gt), head_width=0.05, head_length=0.05, color='green')
        patch2 = patches.FancyArrow(y_bpred, x_bpred, 0.5 * np.cos(phi_bpred), 0.5 * np.sin(
            phi_bpred), head_width=0.05, head_length=0.05, color='blue')
        patch3 = patches.FancyArrow(y_opred, x_opred, 0.5 * np.cos(phi_opred), 0.5 * np.sin(
            phi_opred), head_width=0.05, head_length=0.05, color='red')
        ax1.add_patch(patch1)
        ax1.add_patch(patch2)
        ax1.add_patch(patch3)

        scatter2gthead.set_data(0.02, z_gt)
        scatter2bpredhead.set_data(-0.02, z_bpred)
        scatter2opredhead.set_data(-0.02, z_opred)

        frame = frames[id].astype(np.uint8)
        if isGray == False:
            frame = frame.transpose(1, 2, 0)
        imgplot.set_array(frame)

        annotation2.set_text('Frame {}'.format(id))

        # note: use the first one for viz on screen and second one for video recording
        # return plot1gt, plot1bpr, plot1opr, patch1, patch2, patch3, scatter2gthead, scatter2bpredhead, scatter2opredhead, imgplot, ax1, ax3, annotation, annotation2
        return plot1gt, plot1bpr, plot1opr, patch1, patch2, patch3, scatter2gthead, scatter2bpredhead, scatter2opredhead, imgplot, annotation, annotation2

    ani = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=1, blit=True)
    ani.save(name + '.mp4', writer=writer)
    plt.show()


def main():
    test_set = pd.read_pickle(
        './drone/testdata/160x96PaperTestsetPrune2.pickle')
    h = int(test_set['h'].values[0])
    w = int(test_set['w'].values[0])
    c = int(test_set['c'].values[0])

    x_test = test_set['x'].values
    x_test = np.vstack(x_test[:]).astype(np.float32)
    x_test = np.reshape(x_test, (-1, h, w, c))
    x_test = np.reshape(x_test, (-1, h, w))

    y_test = test_set['y'].values
    y_test = np.vstack(y_test[:]).astype(np.float32)

    device = 'cpu'
    videoname = './out/drone_comparison'

    bmodel = PenguiNet(ConvBlock, [1, 1, 1], True).to(device)
    bmodel.load_state_dict(torch.load(
        './drone/model/model_no_r2/checkpoints/best.pth'))
    bmodel.eval()

    omodel = PenguiNet(ConvBlock, [1, 1, 1], True).to(device)
    omodel.load_state_dict(torch.load(
        './drone/model/model_o_1e0_r2/checkpoints/best.pth'))
    omodel.eval()

    model_input = torch.tensor(x_test)[:, None, ...].to(device)

    bpred = bmodel(model_input).detach()
    opred = omodel(model_input).detach()

    # rolling mean to smooth
    w = 5
    opred = opred.unfold(0, w, 1).mean(-1)
    bpred = bpred.unfold(0, w, 1).mean(-1)
    l = opred.size(0)

    x_test = x_test[:l]
    y_test = y_test[:l]

    print('generating video...')
    VizDroneBEV(x_test, y_test, bpred.numpy(), opred.numpy(),
                videoname, isGray=True)


if __name__ == '__main__':
    main()
