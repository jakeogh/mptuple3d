#!/usr/bin/env python3
# tab-width:4

from __future__ import annotations

import sys

import click
import numpy as np
from asserttool import ic
from click_auto_help import AHGroup
from clicktool import CONTEXT_SETTINGS
from clicktool import click_add_options
from clicktool import click_global_options
from clicktool import tvicgvd
from configtool import get_config_directory
from eprint import eprint
from globalverbose import gvd
from PyQt6.QtWidgets import QApplication  # pylint: disable=E0611
from timestamptool import get_timestamp

from .mptuple3d import enable_dark_mode
from .PointCloud2DViewerDatoviz import PointCloud2DViewerDatoviz
from .PointCloud2DViewerMatplotlib import PointCloud2DViewerMatplotlib
from .PointCloud2DViewerPolyscope import PointCloud2DViewerPolyscope
from .PointCloud2DViewerVispy import PointCloud2DViewerVispy
from .PointCloud3DViewerMatplotlib import PointCloud3DViewerMatplotlib
from .PointCloud3DViewerPolyscope import PointCloud3DViewerPolyscope
from .PointCloud3DViewerVispy import PointCloud3DViewerVispy
from .utils import center_points
from .utils import center_points_2d
from .utils import load_points_from_stdin_for_2d
from .utils import load_points_from_stdin_for_3d
from .utils import load_points_from_stdin_ndarray
from .utils import normalize_points
from .utils import normalize_points_2d

APP_NAME = "mptuple3d"


@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True, cls=AHGroup)
@click_add_options(click_global_options)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    tty, verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )
    config_directory = get_config_directory(click_instance=click, app_name=APP_NAME)
    config_directory.mkdir(exist_ok=True)
    ctx.obj["config_directory"] = config_directory


@cli.command("plot3d-vispy")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable automatic normalization of point cloud coordinates (still centers at origin)",
)
@click.option(
    "--draw-lines",
    is_flag=True,
    help="Draw lines connecting the points in order",
)
@click.option("--xy", is_flag=True, help="Orthographic XY view")
@click.option("--xz", is_flag=True, help="Orthographic XZ view")
@click.option("--yz", is_flag=True, help="Orthographic YZ view")
@click.option(
    "--size",
    type=float,
    help="Point size (default: auto-detected based on point count)",
)
@click.option(
    "--disable-antialiasing",
    is_flag=True,
    help="Disable antialiasing for better performance with large datasets",
)
@click_add_options(click_global_options)
@click.pass_context
def plot3d_vispy(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    draw_lines: bool,
    xy: bool,
    xz: bool,
    yz: bool,
    size: float | None,
    disable_antialiasing: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    3D plot: reads (x,y,z[,color]) tuples from stdin via messagepack.
    """

    tty, verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )
    view_mode = None
    if sum([xy, xz, yz]) > 1:
        print("[ERROR] Only one of --xy, --xz, or --yz may be specified.")
        sys.exit(1)
    if xy:
        view_mode = "xy"
    elif xz:
        view_mode = "xz"
    elif yz:
        view_mode = "yz"

    print("[INFO] Reading points from stdin...")
    points, color_data = load_points_from_stdin_for_3d()

    if points.shape[0] == 0:
        print("[ERROR] No valid points loaded. Exiting.")
        sys.exit(1)

    if disable_normalize:
        print(
            "[INFO] Normalization disabled - centering at origin but preserving scale"
        )
    else:
        print("[INFO] Normalizing points to unit cube")
    app_qt = QApplication(sys.argv)
    enable_dark_mode(app_qt)
    viewer = PointCloud3DViewerVispy(
        points,
        color_data=color_data,
        normalize=not disable_normalize,
        view_mode=view_mode,
        disable_antialiasing=disable_antialiasing,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.show_gui()


cli.add_command(plot3d_vispy, "plot3d")


@cli.command(name="plot2d-vispy")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable normalization (center only).",
)
@click.option(
    "--draw-lines",
    is_flag=True,
    help="Draw lines connecting the points in order",
)
@click.option("--size", type=float, help="Point size")
@click.option("--disable-antialiasing", is_flag=True, help="Disable antialiasing")
@click_add_options(click_global_options)
@click.pass_context
def plot2d_vispy(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    draw_lines: bool,
    size: float | None,
    disable_antialiasing: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    2D plot: reads (x,y[,color]) tuples from stdin via messagepack.
    Preserves X/Y scaling keyboard controls (X, Shift+X, Y, Shift+Y).
    """
    _tty, _verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    print("[INFO] Reading 2D points from stdin...")
    points_xy, color_data = load_points_from_stdin_for_2d()

    if points_xy.shape[0] == 0:
        print("[ERROR] No valid 2D points loaded. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Normalizing points to unit square"
        if not disable_normalize
        else "[INFO] Normalization disabled - centering only"
    )
    app_qt = QApplication(sys.argv)
    enable_dark_mode(app_qt)
    viewer = PointCloud2DViewerVispy(
        points_xy,
        color_data=color_data,
        normalize=not disable_normalize,
        disable_antialiasing=disable_antialiasing,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.show_gui()


@cli.command(name="plot2d-matplotlib")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--normalize",
    is_flag=True,
    help="Enables normalization",
)
@click.option(
    "--draw-lines",
    is_flag=True,
    help="Draw lines connecting the points in order",
)
@click.option("--size", type=float, help="Point size")
@click.option("--disable-antialiasing", is_flag=True, help="Disable antialiasing")
@click_add_options(click_global_options)
@click.pass_context
def plot2d_matplotlib(
    ctx: click.Context,
    keys: tuple[str, ...],
    normalize: bool,
    draw_lines: bool,
    size: float | None,
    disable_antialiasing: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    2D plot (Matplotlib): reads (x,y[,color]) tuples from stdin via messagepack.
    """
    _tty, _verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    print("[INFO] Reading 2D points from stdin...")
    points_xyz = load_points_from_stdin_ndarray(minimum_dimensions=2)

    if points_xyz.shape[0] == 0:
        print("[ERROR] No valid 2D points loaded. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Normalizing points to unit square"
        if normalize
        else "[INFO] Normalization disabled - centering only"
    )
    app_qt = QApplication(sys.argv)
    enable_dark_mode(app_qt)
    viewer = PointCloud2DViewerMatplotlib(
        points_xyz,
        normalize=normalize,
        disable_antialiasing=disable_antialiasing,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.show_gui()


cli.add_command(plot2d_matplotlib, "plot2d")


@cli.command(name="plot2d-datoviz")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable normalization (still frames to data bounds).",
)
@click.option(
    "--draw-lines", is_flag=True, help="Draw lines connecting the points in order"
)
@click.option("--size", type=float, help="Point size (pixels)")
@click_add_options(click_global_options)
@click.pass_context
def plot2d_dvz(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    draw_lines: bool,
    size: float | None,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    2D plot (Datoviz): reads (x,y[,color]) tuples from stdin via messagepack.
    """
    _tty, _verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    print("[INFO] Reading 2D points from stdin...")
    points_xy, color_data = load_points_from_stdin_for_2d()

    if points_xy.shape[0] == 0:
        print("[ERROR] No valid 2D points loaded. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Framing axes to data bounds"
        if disable_normalize
        else "[INFO] Framing axes to padded data bounds"
    )

    # Launch Datoviz viewer
    viewer = PointCloud2DViewerDatoviz(
        points_xy=points_xy,
        color_data=color_data,
        normalize=not disable_normalize,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.run()


@cli.command(name="plot3d-polyscope")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable normalization (center only).",
)
@click.option("--size", type=float, help="Point size (not applicable for Polyscope)")
@click.option(
    "--title",
    type=str,
    default="mptuple3d: 3D (polyscope)",
    help="Window title",
)
@click.option(
    "--cloud-name",
    type=str,
    default="points3d",
    help="Point cloud name",
)
@click.option("--enable-floor", is_flag=True, help="Enable floor grid")
@click_add_options(click_global_options)
@click.pass_context
def plot3d_ps(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    size: float | None,
    title: str,
    cloud_name: str,
    enable_floor: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    3D plot (Polyscope): reads (x,y,z[,color]) tuples from stdin via messagepack.
    """
    _tty, _verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    print("[INFO] Reading 3D points from stdin...")
    points, color_data = load_points_from_stdin_for_3d()

    if points.shape[0] == 0:
        print("[ERROR] No valid 3D points loaded. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Normalizing points"
        if not disable_normalize
        else "[INFO] Normalization disabled"
    )

    # Create Polyscope viewer
    viewer = PointCloud3DViewerPolyscope(
        title=title,
        cloud_name=cloud_name,
        enable_floor=enable_floor,
    )

    # Set points (this will apply normalization if enabled) - use consolidated functions
    if disable_normalize:
        processed_points = center_points(points)
    else:
        processed_points = normalize_points(points)

    viewer.set_points(processed_points)

    # Set colors if available
    if color_data is not None:
        # Convert scalar color data to RGB
        cmin, cmax = color_data.min(), color_data.max()
        if cmax > cmin:
            normalized_colors = (color_data - cmin) / (cmax - cmin)
        else:
            normalized_colors = np.zeros_like(color_data)

        # Simple colormap: blue to red
        colors = np.zeros((len(normalized_colors), 3), dtype=np.float32)
        colors[:, 0] = normalized_colors  # Red channel
        colors[:, 2] = 1.0 - normalized_colors  # Blue channel
        viewer.set_colors(colors)
        viewer.set_scalars("color_data", color_data)

    viewer.show()


@cli.command(name="plot2d-polyscope")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable normalization (center only).",
)
@click.option("--size", type=float, help="Point size (not applicable for Polyscope)")
@click.option(
    "--title",
    type=str,
    default="mptuple3d: 2D (polyscope)",
    help="Window title",
)
@click.option(
    "--cloud-name",
    type=str,
    default="points2d",
    help="Point cloud name",
)
@click.option("--enable-floor", is_flag=True, help="Enable floor grid")
@click_add_options(click_global_options)
@click.pass_context
def plot2d_ps(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    size: float | None,
    title: str,
    cloud_name: str,
    enable_floor: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    2D plot (Polyscope): reads (x,y[,color]) tuples from stdin via messagepack.
    """
    _tty, _verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )

    print("[INFO] Reading 2D points from stdin...")
    points_xy, color_data = load_points_from_stdin_for_2d()

    if points_xy.shape[0] == 0:
        print("[ERROR] No valid 2D points loaded. Exiting.")
        sys.exit(1)

    print(
        "[INFO] Normalizing points"
        if not disable_normalize
        else "[INFO] Normalization disabled"
    )

    # Create Polyscope viewer
    viewer = PointCloud2DViewerPolyscope(
        title=title,
        cloud_name=cloud_name,
        enable_floor=enable_floor,
    )

    # Set points (this will apply normalization if enabled) - use consolidated functions
    if disable_normalize:
        processed_points = center_points_2d(points_xy)
    else:
        processed_points = normalize_points_2d(points_xy)

    viewer.set_points(processed_points)

    # Set colors if available
    if color_data is not None:
        # Convert scalar color data to RGB
        cmin, cmax = color_data.min(), color_data.max()
        if cmax > cmin:
            normalized_colors = (color_data - cmin) / (cmax - cmin)
        else:
            normalized_colors = np.zeros_like(color_data)

        # Simple colormap: blue to red
        colors = np.zeros((len(normalized_colors), 3), dtype=np.float32)
        colors[:, 0] = normalized_colors  # Red channel
        colors[:, 2] = 1.0 - normalized_colors  # Blue channel
        viewer.set_colors(colors)
        viewer.set_scalars("color_data", color_data)

    viewer.show()


@cli.command(name="plot3d-matplotlib")
@click.argument("keys", type=str, nargs=-1)
@click.option(
    "--disable-normalize",
    is_flag=True,
    help="Disable automatic normalization of point cloud coordinates (still centers at origin)",
)
@click.option(
    "--draw-lines",
    is_flag=True,
    help="Draw lines connecting the points in order",
)
@click.option("--xy", is_flag=True, help="Orthographic XY view")
@click.option("--xz", is_flag=True, help="Orthographic XZ view")
@click.option("--yz", is_flag=True, help="Orthographic YZ view")
@click.option(
    "--size",
    type=float,
    help="Point size (default: auto-detected based on point count)",
)
@click.option(
    "--disable-antialiasing",
    is_flag=True,
    help="Disable antialiasing for better performance with large datasets",
)
@click_add_options(click_global_options)
@click.pass_context
def plot3d_mpl(
    ctx: click.Context,
    keys: tuple[str, ...],
    disable_normalize: bool,
    draw_lines: bool,
    xy: bool,
    xz: bool,
    yz: bool,
    size: float | None,
    disable_antialiasing: bool,
    verbose_inf: bool,
    dict_output: bool,
    verbose: bool = False,
) -> None:
    """
    3D plot (Matplotlib): reads (x,y,z[,color]) tuples from stdin via messagepack.
    Features interactive 3D visualization with mouse controls and keyboard scaling.
    """

    tty, verbose = tvicgvd(
        ctx=ctx,
        verbose=verbose,
        verbose_inf=verbose_inf,
        ic=ic,
        gvd=gvd,
    )
    view_mode = None
    if sum([xy, xz, yz]) > 1:
        print("[ERROR] Only one of --xy, --xz, or --yz may be specified.")
        sys.exit(1)
    if xy:
        view_mode = "xy"
    elif xz:
        view_mode = "xz"
    elif yz:
        view_mode = "yz"

    print("[INFO] Reading points from stdin...")
    points, color_data = load_points_from_stdin_for_3d()

    if points.shape[0] == 0:
        print("[ERROR] No valid points loaded. Exiting.")
        sys.exit(1)

    if disable_normalize:
        print(
            "[INFO] Normalization disabled - centering at origin but preserving scale"
        )
    else:
        print("[INFO] Normalizing points to unit cube")

    app_qt = QApplication(sys.argv)
    enable_dark_mode(app_qt)
    viewer = PointCloud3DViewerMatplotlib(
        points,
        color_data=color_data,
        normalize=not disable_normalize,
        view_mode=view_mode,
        disable_antialiasing=disable_antialiasing,
        draw_lines=draw_lines,
        size=size,
    )
    viewer.show_gui()
