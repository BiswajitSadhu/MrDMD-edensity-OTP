import pandas as pd
import numpy as np
from pydmd import MrDMD
from pydmd import DMD
import matplotlib.pyplot as plt

def plot_eigs_path(self,path, system, show_axes=True,show_unit_circle=True,figsize=(8, 8),title='',level=None,node=None):
        """
        Plot the eigenvalues.

        :param bool show_axes: if True, the axes will be showed in the plot.
                Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
                and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches of the figure.
        :param str title: title of the plot.
        :param int level: plot only the eigenvalues of specific level.
        :param int node: plot only the eigenvalues of specific node.
        
        """
        import matplotlib.font_manager as font_manager
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns  # for nicer graphics

        matplotlib.rcParams.update({'font.size': 20})
        matplotlib.rcParams['font.family'] = ['DejaVu Serif']
        font = font_manager.FontProperties(family='DejaVu Serif',
                                   weight='normal',
                                   style='normal', size=20)

        matplotlib.rc('axes', linewidth=2)
        
        if self.eigs is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')

        if level:
            peigs = self.partial_eigs(level=level, node=node)
        else:
            peigs = self.eigs

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        if not level:
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i) for i in np.linspace(0, 1, len(self.dmd_tree.levels)-1)]

            points = []
            for level_num in range(len(self.dmd_tree.levels)-1):
                eigs = self.partial_eigs(level_num)

                points.append(
                    ax.plot(eigs.real, eigs.imag, '*', linewidth=3, markersize=12, color=colors[level_num])[0])
        else:
            points = []
            points.append(
                ax.plot(peigs.real, peigs.imag, '*', linewidth=3, markersize=12, label='Eigenvalues')[0])

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(peigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        plt.ylabel('Imaginary part')
        plt.xlabel('Real part')


        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-.')
        ax.grid(True)

        ax.set_aspect('equal')

        # x and y axes
        if show_axes:
            ax.annotate(
                '',
                xy=(np.max([limit * 0.8, 1.]), 0.),
                xytext=(np.min([-limit * 0.8, -1.]), 0.),
                arrowprops=dict(arrowstyle="->"))
            ax.annotate(
                '',
                xy=(0., np.max([limit * 0.8, 1.])),
                xytext=(0., np.min([-limit * 0.8, -1.])),
                arrowprops=dict(arrowstyle="->"))

        # legend
        if level:
            labels = ['Eigenvalues - level {}'.format(level)]
        else:
            labels = [
                'Eigenvalues - level {}'.format(i+1)
                for i in range(self.max_level)
            ]

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0., 0.), 1., color='black', fill=False, linestyle='--',linewidth=2)
            ax.add_artist(unit_circle)
            points += [unit_circle]
            labels += ['Unit circle']

        ax.add_artist(plt.legend(points, labels, loc='best'))
        plt.savefig('%s/eigan_val_in_complex_plane_%s' %(path,system), dpi=600)

