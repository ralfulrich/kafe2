from kafe2 import HistContainer, HistFit, HistPlot, ContoursProfiler
import matplotlib.pyplot as plt


def leg_poly1(x, a=100, b=1.5):
    return a + b * x


def legendre_grade_2(x, a=1., b=1., c=0.):
    return a + b * x + c * 0.5 * (3 * x ** 2 - 1)


def legendre_grade_2_integrated(x, a, b, c):
    return 0.5 * x * (2 * a + b * x + c * (x ** 2 - 1))


data = [110, 123, 145, 140, 164, 188, 205, 216, 226, 194, 210, 205, 204, 187]
n_bins = 14
bin_range = (109, 151)
bins = [109., 112., 115., 118., 121., 124., 127., 130., 133., 136., 139.,  142., 145., 148., 151.]
if __name__ == '__main__':
    hist_data = HistContainer(n_bins=n_bins, bin_range=bin_range, bin_edges=bins)
    hist_data.set_bins(data)
    #hist_fit_1 = HistFit(hist_data, model_density_function=leg_poly1)
    #hist_fit_1.do_fit()
    hist_fit_2 = HistFit(hist_data, model_density_function=legendre_grade_2)
    hist_fit_2.do_fit()
    plot = HistPlot(hist_fit_2)
    plot.show_fit_info_box()
    plot.plot()
    cpf = ContoursProfiler(hist_fit_2)
    cpf.plot_profiles_contours_matrix()
    plt.show()
