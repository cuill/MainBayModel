from __future__ import annotations

import logging.config
import time

import holoviews as hv
import panel as pn
from holoviews import opts as hvopts

import chesbay

# load bokeh
hv.extension("bokeh")
pn.extension(sizing_mode="scale_width")

# Set some defaults for the visualization of the graphs
hvopts.defaults(
    hvopts.Image(  # pylint: disable=no-member
        # Don't set both height and width, or the UI will not be responsive!
        #width=800,
        #height=500,
        responsive=False,
        frame_width=800,
        frame_height=600,
        show_title=True,
        tools=["hover"],
        active_tools=["pan", "wheel_zoom"],
        align="end",
    ),
    hvopts.Layout(toolbar="right"),  # pylint: disable=no-member
)

#use template
# https://panel.holoviz.org/reference/templates/Bootstrap.html

bootstrap = pn.template.BootstrapTemplate(
    header_background='lightblue',
    title="Main Bay Model",
    logo="chesbay/assets/VIMS_CCRM_logo.jpg",
    favicon="chesbay/assets/SCHISM-logo.png",
    sidebar_width=350,  # in pixels! must be an integer!
)

#call UI
ui = chesbay.ChesBayUI(app_bootstrap=bootstrap, display_stations=False)

bootstrap.main.append(ui.main)
bootstrap.sidebar.append(ui.sidebar)
bootstrap.modal.append(ui.modal)

bootstrap.servable()
