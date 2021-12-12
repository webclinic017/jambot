from typing import *

import pandas as pd
from seaborn import diverging_palette

_cmap = diverging_palette(240, 10, sep=10, n=21, as_cmap=True, center='dark')


def format_cell(bg, t='black'):
    return f'background-color: {bg};color: {t};'


def bg(style, subset=None, higher_better=True, axis=0):
    """Show style with highlights per column"""
    if subset is None:
        subset = style.data.columns

    cmap = _cmap.reversed() if higher_better else _cmap

    return style \
        .background_gradient(cmap=cmap, subset=subset, axis=axis)


def highlight_val(df, m: dict):

    m_replace = {k: format_cell(bg=v[0], t=v[1]) for k, v in m.items()}

    return df.replace(m_replace)


def get_style(df):
    return df.style \
        .format('{:.3f}')


def background_grad_center(
        s: pd.Series,
        center: float = 0,
        vmin: float = None,
        vmax: float = None,
        higher_better: bool = True) -> Union[list, pd.DataFrame]:
    """Style column with diverging color palette centered at value, including dark/light text formatting
    - modified from https://github.com/pandas-dev/pandas\
    /blob/b7cb3dc25a5439995d2915171c7d5172836c634e/pandas/io/formats/style.py

    Parameters
    ----------
    s : pd.Series
        Column to style
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        default self.cmap
    center : int, optional
        value to center diverging color, default 0
    vmin : float, optional
        min value, by default None
    vmax : float, optional
        max value, by default None

    Returns
    -------
    Union[list, pd.DataFrame]
        list of background colors for styler or df if multiple columns
    """
    from matplotlib.colors import TwoSlopeNorm, rgb2hex
    cmap = _cmap.reversed() if higher_better else _cmap  # type: ignore

    vmin = vmin or s.values.min()  # type: float
    vmax = vmax or s.values.max()  # type: float

    # vmin/vmax have to be outside center
    if vmin >= center:
        vmin = center - 0.01

    if vmax <= center:
        vmax = center + 0.01

    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    text_color_threshold = 0.408  # default from pandas

    def relative_luminance(rgba) -> float:
        """Check if rgba color is greater than darkness threshold"""
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3])

        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba) -> str:
        dark = relative_luminance(rgba) < text_color_threshold
        text_color = '#f1f1f1' if dark else '#000000'
        return f'background-color: {rgb2hex(rgba)}; color: {text_color};'

    rgbas = cmap(norm(s.astype(float).values))

    if s.ndim == 1:
        return [css(rgba) for rgba in rgbas]
    else:
        return pd.DataFrame(
            [[css(rgba) for rgba in row] for row in rgbas],
            index=s.index,
            columns=s.columns)
