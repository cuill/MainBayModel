# pylint: disable=unused-argument,no-member
from __future__ import annotations

import time
import glob
import logging
import os.path
from datetime import datetime

import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
import geoviews as gv
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize
from pyproj import Transformer, Proj, transform
from holoviews.streams import PointerXY, DoubleTap

from . import utils

logger = logging.getLogger(__name__)
DATA_DIR = "data" + os.path.sep + "*"

# CSS Styles
ERROR = {"border": "3px solid red"}
INFO = {"border": "2px solid blue"}

# Help functions that log messages on stdout AND render them on the browser
def info(msg: str) -> pn.Column:
    logger.info(msg)
    return pn.pane.Markdown(msg, style=INFO)

def error(msg: str) -> pn.Column:
    logger.error(msg)
    return pn.pane.Markdown(msg, style=ERROR)


class ChesBayUI:  # pylint: disable=too-many-instance-attributes
    """
    This UI is supposed to be used with a Bootstrap-like template supporting
    a "main" and a "sidebar":

    - `sidebar` will contain the widgets that control what will be rendered in the main area.
      E.g. things like which `source_file` to use, which timestamp to render etc.

    - `main` will contain the rendered graphs.

    - `modal`: a modal is a pop-up area.

    In a nutshell, an instance of the `UserInteface` class will have two private attributes:

    - `_main`
    - `_sidebar`

    These objects should be of `pn.Column` type. You can append
    """

    def __init__( self, app_bootstrap = None, display_stations: bool = False) -> None:
        self._bootstrap = app_bootstrap
        self._display_stations = display_stations

        #UI components
        #self._main = pn.Column(error("## Please select a `dataset_file` and click on the `Render` button."))
        #This is to show grid when loading the page, an alternative way is to show project description.
        ds = xr.open_dataset('data/schout_1.nc')
        self._main = pn.Column(utils.plot_grid(ds, 'epsg:26918'))

        self._sidebar = pn.Column()
        self._modal = pn.Column()
        self._bootstrap.modal.append(pn.Column())

        #Define widgets  
        self.dataset_file      = pn.widgets.Select(
            name="Dataset file", options=sorted(filter(utils.can_be_opened_by_xarray, glob.glob(DATA_DIR))),
        )
        self.dataset_format    = pn.widgets.Select(name="Format", options=["netCDF", "GeoJSON"])
        self.prj               = pn.widgets.Select(options=["epsg:4326", "epsg:26918"], value='epsg:26918', name="Projection")
        self.variable          = pn.widgets.Select(name="Variable")
        self.layer             = pn.widgets.Select(name="Layer", options=["surface","bottom"], value="surface")

        #time series (True/False)
        self.timeseries        = pn.widgets.Checkbox(name="Time Series (click on the map)")

        #stations
        self.stations_file     = pn.widgets.Select(name="Stations file")
        self.stations          = pn.widgets.CrossSelector(name="Stations")

        # render button
        self.render_button     = pn.widgets.Button(name="Render", button_type="primary")

        #about button
        self.about_button      = pn.widgets.Button(name="About", button_type="primary")


        self._define_widget_callbacks()
        self._populate_widgets()
        self._setup_ui()

    def _about_callback(self, event: pn.Event):
        self._bootstrap.modal[0].clear()
        #self._bootstrap.modal[0].objects = [pn.pane.Markdown('This is a test app!', style=INFO)]
        self._bootstrap.modal[0].objects = [info('Here is May Bay Model description!')]
        #time.sleep(10)
        self._bootstrap.open_modal()

    def _error_info(self, event: pn.Event):
        if self.dataset_format.value != 'netCDF':
            self._bootstrap.modal[0].clear()
            self._bootstrap.modal[0].objects = [info('This feature is not implemented yet! Please choose other options.')]
            self._bootstrap.open_modal()

    def _setup_ui(self) -> None:

        self._sidebar.append(
            pn.Accordion(
                ("Input Files", pn.WidgetBox(self.dataset_file, pn.Row(self.dataset_format,self.prj),
                 pn.Row(self.variable,self.layer))),
                active=[0],
            ),
        )

        #Render button to show animation
        self._sidebar.append(self.render_button)

        self._sidebar.append(
            pn.Accordion(
                ("TimeSeries", pn.WidgetBox(self.timeseries)),
                active=[0],
            ),
        )

        if self._display_stations:
            self._sidebar.append(
                pn.Accordion(("Stations", pn.WidgetBox(self.stations_file, self.stations))),
            )

        #About button to show project description
        self._sidebar.append(self.about_button)

    def _define_widget_callbacks(self) -> None:
        #Dataset callback
        self.dataset_file.param.watch(fn=self._read_header_info, parameter_names="value")
        self.prj.param.watch(fn=self._read_header_info, parameter_names="value")
        self.dataset_format.param.watch(fn=self._error_info, parameter_names="value")

        #Render button
        self.render_button.on_click(self._animation)

        #timeseries callback
        self.timeseries.param.watch(fn=self._timeseries, parameter_names="value")

        #Station callbacks (todo)

        #About button
        self.about_button.on_click(self._about_callback)

    def _populate_widgets(self) -> None:
        self.dataset_file.param.trigger("value")


    def _read_header_info(self, event: pn.Event): 
        self._filename  = self.dataset_file.value
        self._format    = self.dataset_format.value
        self._prj       = self.prj.value
        self._layer     = self.layer.value

        hdata           = utils.read_dataset(self._filename, 1,self._format, self._prj)
        self._dataset   = hdata[0]
        self._times     = hdata[1]
        #self._variables = hdata[2]
        self._x         = hdata[3]
        self._y         = hdata[4]
        self._z         = hdata[5]
        self._elnode    = hdata[6]
        
        #transform projection
        if self._prj != 'epsg:4326':
           self._x, self._y = Transformer.from_crs(self._prj, 'epsg:4326', always_xy=True).transform(self._x, self._y)
        
        self.variable.param.set_param(options=hdata[2], value=hdata[2][0])

    def _debug_ui(self) -> None:
        logger.debug("Widget values:")
        widgets = [obj for (name, obj) in self.__dict__.items() if isinstance(obj, pn.widgets.Widget)]
        for widget in widgets:
            logger.debug("%s: %s", widget.name, widget.value)

    def _animation(self, event: pn.Event):

        #set timeseries to False when reloading
        self.timeseries.value = False
        
        #read data
        ds, times = utils.read_dataset(self._filename, method=4, 
            variable = self.variable.value, layer = self.layer.value)

        df = pd.DataFrame({'longitude': self._x, 'latitude': self._y, 'data': ds[0, :]}) 
        pdf = gv.Points(df, kdims = ['longitude', 'latitude'], vdims = 'data')
        opts.defaults(opts.WMTS(width=1200, height=900))

        def time_mesh(time):

            #Get slab at selected time
            tid = np.nonzero(times == time)[0].item()
            pdf.data.data = ds[tid, :]

            return gv.TriMesh((self._elnode, pdf))

        trimesh = hv.DynamicMap(time_mesh, kdims='Time').redim.values(Time=times)
        trimap = rasterize(trimesh, precompute=True).opts(
            title=self.variable.value,
            colorbar = True,
            cmap = 'jet',
            show_legend = True,
            )

        t_widget = pn.widgets.Select()
        tiles = utils.get_tiles()

        @pn.depends(t_widget)
        def t_plot(time):
            return tiles * trimap

        self._main.objects = [t_plot]

    def _timeseries(self, event: pn.Event):

        #Declare trimap 
        trimap = utils.plot_grid(self._dataset, self._prj, False)

        #Declare Tap stream with trimap as source 
        posxy = hv.streams.Tap(source=trimap, x=-75.77, y=36.9638, transient=True)

        #Define function to get timeseries data based on tap location (nearest points),
        #therefore, if the location is out of the domain, it still returns values.
        def tap_map(x, y):

            #Find the nearest node id
            dist = abs(self._x + 1j * self._y - x - 1j * y)
            node = np.nonzero(dist == dist.min())[0][0]

            #Get the timeseries data for node
            ts_data = utils.read_dataset(self.dataset_file.value, 3, 
                self.dataset_format.value, variable=self.variable.value, 
                layer = self.layer.value, node = node)

            curve = hv.Curve((self._times, ts_data),'time',self.variable.value).opts(
                color='k', line_width=2, tools=["hover"]
            )
            return curve

        #Connect the Tap stream to the trimap callback
        hcurve = hv.DynamicMap(tap_map, streams=[posxy]).opts(
            height=300, framewise=True, responsive=True
        )
 
        #Display the mesh and Curve side by side
        hmap = (trimap + hcurve) 

        #Update _main
        if (len(self._main)) == 1:
           self._main.append(hmap)
        else:
           self._main[1].objects = [hmap]

    @property
    def sidebar(self) -> pn.Column:
        return self._sidebar

    @property
    def main(self) -> pn.Column:
        return self._main

    @property
    def modal(self) -> pn.Column:
        return self._modal
