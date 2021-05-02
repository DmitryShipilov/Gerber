import sys
import numpy as np
from matplotlib import pyplot
from matplotlib.widgets import Button,CheckButtons
import gerber
import gerber_shapely as gs

import pandas as pd
import random

import xml.etree.ElementTree as et

INCH = 25.4 #mm
MIL = INCH/1000.0

convname="_conv"
class Project:
	def __init__(self,config_file):
		self.config_file=config_file
		self.layers=[]
		self.code_start=""
		self.code_end=""
		self.layer_plot={}
		self.CheckButtonNames=()
		self.CheckButtonBools=()

class Layer:
	def __init__(self, name="",layertype=0):
		self.gerber_dir = ""
		self.gerber_file = ""
		self.gerbere_ext="*.gtl"
		self.name = name
		self.type = layertype
		self.gcode_ext="*.ngc"
		self.gcode_dir = ""
		self.gcode_file = ""
		self.mirror_enable = False
		self.mirror_axis = 0
		self.mirror_px = 0
		self.mirror_py = 0
		self.rot_angle = 0
		self.circle_angle = 30
		self.tool_dia = 1.0
		self.tool_type = 0
		self.tool_circle_ang = 20
		#self.tool_asobi = tool_d/10.0
		self.tool_xy_spee=0
		self.tool_z_spee=0
		self.tool_depth=-1.0
		self.tool_z_step=0.5
		#for minimum cutting
		self.ref_length=0.01
		self.remove_length=0.01

		self.zone_segment = False
		#self.scrape_all=0
		self.scrape_step_r=1.0
		self.scrape_step=1.0*self.tool_dia
		self.scrape_max = 0
		self.scrape_margin_r=1.1

		self.x_offset = 5.0
		self.y_offset  = 5.0
		self.spindle_enable=1
		self.spindle_speed=0
		self.unit_inch=0
		#
		self.draw_color="red"
		self.draw_fill=""
		self.draw_convcolor="red"




class Line:
    def __init__(self, x1, x2, y1, y2, w):
        self.xBegin = x1
        self.xEnd = x2
        self.yBegin = y1
        self.yEnd = y2
        self.width = w

        # y + kx = b
        self.k = (int)(self.yBegin - self.yEnd)
        self.b = (int)(self.yBegin * (self.xEnd - self.xBegin)
                       - self.xBegin * (self.yEnd - self.yBegin))

    def checkAffilation(self, x):
        if x < self.xBegin or x > self.xEnd:
            return False
        return True

def findCross(line1, line2):
    if not isinstance(line1, Line):
        print("First argument is not Line")
        return

    if not isinstance(line2, Line):
        print("Second argument is not Line")
        return

    # y + kx = b
    # y always equals 1
    a = np.array([[1, line1.k], [1, line2.k]])
    b = np.array([line1.b, line2.b])
    result = np.linalg.solve(a, b)
    return result

def crossIsValid(line1, line2, x):
    if not line1.checkAffilation(x):
        return False

    if not line2.checkAffilaition(x):
        return False

    return True


def read_config(prj):
	try:
		tree = et.parse(prj.config_file)
	except IOError:
		print ("Unable to open the file =" + prj.config_file + "\n")
	else:
		prj.layers=[]
		elem = tree.getroot()
		prj.name=elem.attrib['name']
		for child in elem:
			#print (child)
			if child.tag=="defaults":
				for d in child:
					for key, val in d.items():
						if d.tag=="code":
							print ("code")
							if key=="start":
								prj.code_start=val
							else:
								prj.code_end=val
						else:
							exec("prj."+d.tag + "_" + key + "=\""+val+"\"")
			if child.tag=="layers":
				for l in child:
					if l.tag=="layer":
						tmp_layer=Layer(l.get("name"),l.get("type"))
						prj.layers.append(tmp_layer)
						for l_data in l:
							for key, val in l_data.items():
								if val=="":
									exec("tmp_layer."+l_data.tag + "_" + key + " = prj."+l_data.tag + "_" + key)
								else:
									exec("tmp_layer."+l_data.tag + "_" + key + " = \""+val+"\"")

def layer_setting(layer,prj):
	#if layer.rot_angle
	if int(layer.tool_xy_speed) < 1:
		layer.tool_xy_speed=prj.tool_xy_speed
	if int(layer.tool_z_speed) < 1:
		layer.tool_z_speed=int(prj.tool_z_speed)
	if int(layer.spindl_enable) and  int(layer.spindl_speed)<1:
		layer.spindl_speed=int(prj.spindl_speed)
	if float(layer.unit_out) <= 0:
		layer.unit_out=float(prj.unit_out)

def plot_coords(ax, ob,plot_color):
	x, y = ob.xy
	ret, = ax.plot(x, y, '-', color=plot_color, zorder=1)
	return ret

def plot_point(ax, ob,plot_color):
	x, y = ob.xy
	#http://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html
	#ax.plot(x, y, ',', color=plot_color, zorder=1)
	return ax.plot(x, y, '.', color=plot_color, zorder=1)

def PoltUnit(ax,layer,figs,color):
	plots=[]
	for polygon in figs.elements:
		if polygon.active:
			if polygon.element.geom_type == 'Polygon':
				plots.append(plot_coords(ax, polygon.element.exterior,color))
			elif polygon.element.geom_type == 'MultiPolygon':
				for poly in polygon.element:
					plots.append(plot_coords(ax, poly.exterior,color))
			elif polygon.element.geom_type== "LineString":
				plots.append(plot_coords(ax, polygon.element.coords,color))
			elif polygon.element.geom_type== "Point":
				#print "Ponint"
				plots.append(plot_point(ax, polygon.element.coords,color))
	return plots

def FirstPlot(prj):
	for layer in prj.layers:
		layer_setting(layer,prj)
		grb = gerber.Gerber(layer.gerber_dir,layer.gerber_file,float(prj.unit_out))
		prj.CheckButtonNames=prj.CheckButtonNames+(layer.name,)
		prj.CheckButtonBools=prj.CheckButtonBools+(True,)
		prj.CheckButtonNames=prj.CheckButtonNames+(layer.name+convname,)
		prj.CheckButtonBools=prj.CheckButtonBools+(True,)
		if int(layer.type) == 0:	#Cu layer
			if float(layer.scrape_step_r) !=1.0 and float(layer.scrape_step_r) !=0.0:
				layer.scrape_step = float(layer.scrape_step_r)* float(layer.tool_dia)
			tmp_elements = []
			tmp_xmax = 0.0
			tmp_ymax = 0.0
			tmp_xmin = 0.0
			tmp_ymin = 0.0
			op = gs.Gerber_OP(grb,float(layer.tool_dia))
			op.gerber2shapely()
			# На этом этапе внутри op лежат все линии, которые только возможны для текущего слоя
			op.in_unit=grb.in_unit
			prj.layer_plot[layer.name]=PoltUnit(prj.MainPlot,layer,op.raw_figs,layer.draw_color)
		if int(layer.type) == 1:	#Drill layer
			grb=gerber.Drill(layer.gerber_dir,layer.gerber_file,float(prj.unit_out))
			op = gs.Gerber_OP(grb,float(layer.tool_dia))
			op.in_unit=grb.in_unit
			op.out_unit=float(prj.unit_out)
			op.drill2shapely(0)
			prj.layer_plot[layer.name]=PoltUnit(prj.MainPlot,layer,op.raw_figs,layer.draw_color)
		if int(layer.type) == 2:	#Edge layer
			op = gs.Gerber_OP(grb,float(layer.tool_dia))
			op.in_unit=grb.in_unit
			op.out_unit=float(prj.unit_out)
			op.edge2shapely()
			prj.layer_plot[layer.name]=PoltUnit(prj.MainPlot,layer,op.raw_figs,layer.draw_color)
		if int(layer.type) == 3:	#Paste (Stencil mask)
			#continue
			op = gs.Gerber_OP(grb,float(layer.tool_dia))
			op.in_unit=grb.in_unit
			op.out_unit=float(prj.unit_out)
			op.paste2shapely()
			prj.layer_plot[layer.name]=PoltUnit(prj.MainPlot,layer,op.raw_figs,layer.draw_color)

def Conv(prj):
	#prj.MsgWindow.text(0,0.9,"Start Convert")
	#pyplot.draw()
	msgp=-1
	for layer in prj.layers:
		layer_setting(layer,prj)
		grb = gerber.Gerber(layer.gerber_dir,layer.gerber_file,float(prj.unit_out))
		if int(layer.type) == 0:	#Cu layer
			msgp+=1
			if float(layer.scrape_step_r) !=1.0 and float(layer.scrape_step_r) !=0.0:
				layer.scrape_step = float(layer.scrape_step_r)* float(layer.tool_dia)
			tmp_elements = []
			tmp_xmax = 0.0
			tmp_ymax = 0.0
			tmp_xmin = 0.0
			tmp_ymin = 0.0
			if layer.minimum_enable!="0":
				op = gs.Gerber_OP(grb,0.0)
				op.gerber2shapely()
				op.in_unit=grb.in_unit
				op.out_unit=float(prj.unit_out)
				op.mirror=int(layer.mirror_enable)
				op.rot_ang=float(layer.rot_angle)

				op.ref_length=float(layer.minimum_ref_length)
				op.remove_length=float(layer.minimum_remove_length)
				op.get_minmax(op.tmp_figs)
				margin=float(layer.minimum_margin)
				if layer.minimum_type=="0":
					print ("Minimum cutting")
					op.minimum(op.xmax+margin,op.ymax+margin,op.xmin-margin,op.ymin-margin)
				elif layer.minimum_type=="1":
					print ("Voronoi cutting. Heavy...")
					op.voronoi(op.xmax+margin,op.ymax+margin,op.xmin-margin,op.ymin-margin)
				op.get_minmax(op.figs)
				op.affine()
			else:
				for i in range(int(layer.scrape_max)):
					op = gs.Gerber_OP(grb,float(layer.tool_dia)+i*float(layer.scrape_step))
					op.gerber2shapely()
					op.in_unit=grb.in_unit
					#op.out_unit=layer.unit_out
					op.out_unit=float(prj.unit_out)
					op.mirror=int(layer.mirror_enable)
					op.rot_ang=float(layer.rot_angle)

					op.merge_polygon()
					op.get_minmax(op.figs)
					op.affine()
					if i == 0:
						tmp_xmax = op.xmax+float(layer.scrape_margin_r)*float(layer.tool_dia)
						tmp_ymax = op.ymax+float(layer.scrape_margin_r)*float(layer.tool_dia)
						tmp_xmin = op.xmin-float(layer.scrape_margin_r)*float(layer.tool_dia)
						tmp_ymin = op.ymin-float(layer.scrape_margin_r)*float(layer.tool_dia)
						center=op.center
					op.limit_cut(tmp_xmax,tmp_ymax,tmp_xmin,tmp_ymin)
					print ("Loop No. = ",i,", number of plygons = ",len(op.figs.elements))
					tmp_elements += op.figs.elements
					#f_op.figs.elements=[]
					#print "num",len(tmp_elements)
				op.figs.elements=tmp_elements
				op.center = center
			xoff=float(layer.shift_x_offset)
			yoff=float(layer.shift_y_offset)
			op.xoff = xoff
			op.yoff = yoff
			op.affine()
			op.affine_trans(op.raw_figs)
			op.draw_out()
			op.fig_out()
		if int(layer.type) == 2:	#Edge layer
			e_op = gs.Gerber_OP(grb,float(layer.tool_dia))
			e_op.in_unit=grb.in_unit
			e_op.out_unit=float(prj.unit_out)
			e_op.edge2shapely()
			e_op.merge_line()
			e_op.mirror=int(layer.mirror_enable)
			e_op.rot_ang=float(layer.rot_angle)
			e_op.get_minmax(e_op.figs)
			#e_op.center = center
			xoff=float(layer.shift_x_offset)
			yoff=float(layer.shift_y_offset)
			e_op.xoff = xoff
			e_op.yoff = yoff
			e_op.affine()
			e_op.affine_trans(e_op.raw_figs)
			e_op.fig_out()
		if int(layer.type) == 3:	#Paste (Stencil mask)
			print ("Stencil mask")
			#continue
			p_op = gs.Gerber_OP(grb,float(layer.tool_dia))
			p_op.in_unit=grb.in_unit
			p_op.out_unit=float(prj.unit_out)
			p_op.paste2shapely()
			p_op.merge_line()
			p_op.mirror=int(layer.mirror_enable)
			p_op.rot_ang=float(layer.rot_angle)
			p_op.get_minmax(p_op.figs)
			xoff=float(layer.shift_x_offset)
			yoff=float(layer.shift_y_offset)
			p_op.xoff = xoff
			p_op.yoff = yoff
			p_op.affine()
			p_op.affine_trans(p_op.raw_figs)
			p_op.fig_out()

def createExcel():
	curcuits = {}
	max_curuits = random.randint(500, 1000)
	for index in range(max_curuits):
		curcuits[index] = random.uniform(0.1, 5)

	df = pd.DataFrame(data = curcuits)
	df.to_excel('D:\MAI\!Magistrature\Dissertaton\Проект диплома\Gerbrer description\gerber2gcode\python_system_equation')


def MainAxes(prj):
	fig = pyplot.figure(figsize=(14,10))
	fig.canvas.set_window_title('DiagnosticControl')
	fig.suptitle(prj.name, fontsize=10, fontweight='bold')

	####### Create axes ########
	#axes([left, bottom, width, height])
	prj.DispLayer = pyplot.axes([0.05, 0.25, 0.15, 0.7])
	prj.MainPlot = pyplot.axes([0.27, 0.25, 0.7, 0.7])
	prj.MsgWindow = pyplot.axes([0.05, 0.05, 0.7, 0.1])
	prj.Button = pyplot.axes([0.85, 0.05, 0.1, 0.05])
	####### Axes setup ########
	#prj.DispLayer.axis('off')
	prj.MsgWindow.axis('off')
	prj.DispLayer.set_title("Layers")
	prj.MsgWindow.set_title("Messages")
	prj.MainPlot.set_aspect(1)
	#prj.DispLayer.set_axis_bgcolor('w')
	prj.CheckButtonNames=()
	prj.CheckButtonBools=()
	prj.layer_plot={}
	#### Functions ###
	def func(label):
		if label in prj.layer_plot:
			for p in prj.layer_plot[label]:
				if isinstance(p, list):
					#print (p)
					for pp in p:
						pp.set_visible(not pp.get_visible())
					continue
				p.set_visible(not p.get_visible())
			pyplot.draw()
	class Action(object):
		def conv(self, event):
			print ("Calculate start!")
			"""
			prj.MainPlot.cla()
			prj.layer_plot={}
			Conv(prj)
			pyplot.draw()
			"""

			createExcel()

			print("DONE")

	unit = "mm"
	if prj.unit_out == 25.4:
		unit = "inch"
	prj.MainPlot.set_xlabel(unit)
	prj.MainPlot.set_ylabel(unit)
	########
	FirstPlot(prj)
	########
	check = CheckButtons(prj.DispLayer, prj.CheckButtonNames, prj.CheckButtonBools)
	check.on_clicked(func)
	callback = Action()
	bconv = Button(prj.Button, 'Calculate voltage')
	bconv.on_clicked(callback.conv)
	pyplot.show()


def main():
    """
    x1 = 0
    y1 = 0
    w1 = 1

    x2 = 2
    y2 = 2
    w2 = 1

    line1 = Line(x1, x2, y1, y2, w1)
    line2 = Line(-x1, -x2, -y1, -y2, w2)

    result = findCross(line1, line2)
    valid = crossIsValid(line1, line2, result[0])

    a = 3
"""
    #plt.plot()
    #plt.show()

    if len(sys.argv) > 1 and sys.argv[1]:
        # read_config(sys.argv[1])
        prj = Project(sys.argv[1])
        read_config(prj)
    else:
        print("Usage : python pygerber2gcode_cui.py config_file")
        exit()

    if len(prj.layers) < 1:
        print("Error:There are no layers")
        exit()
    MainAxes(prj)

if __name__ == "__main__":
	main()