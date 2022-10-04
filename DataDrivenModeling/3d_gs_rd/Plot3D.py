import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.io

def postprocess3D(data, isU=True, resFlag = 0, num=None):
    x = np.linspace(-50, 50, 48)
    y = np.linspace(-50, 50, 48)
    z = np.linspace(-50, 50, 48)
    x, y, z = np.meshgrid(x, y, z)

    appd = ['PeRCNN' 'Truth']
    uv = ['v', 'u']
    values = data

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0.3 if isU else 0.1,
        isomax=0.5 if isU else 0.3,
        opacity=0.2,
        colorscale='RdBu', # 'BlueRed',
        surface_count=2,  # number of isosurfaces, 2 by default: only min and max
    ))
    # fig.show()
    fig.write_image('./figures/Iso_surf_%s_%s_%d.png' % (uv[isU], appd[resFlag], num))
    plt.close('all')


if __name__ == "__main__":

    # grid size
    N = 48

    data = scipy.io.loadmat('./uv_2x31x48x48x48_[PeRCNN].mat')['uv']     # 10x downsampled in time dim
    for i in range(0, 31,  2):
        postprocess3D(data[0, i], isU=True, num=i)