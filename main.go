package main

import (
	"encoding/csv"
	"github.com/kniren/gota/dataframe"
	"github.com/sajari/regression"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"math"
	"os"
	"strconv"
	"time"
)

func main() {
	log.Println("teste computatção grafica ")
	log.Println(time.Now())
	var caminho string
	caminho  =   "Machine-Learning-with-R-datasets-master/"
	file,  err  :=os.Open(caminho+"concrete.csv")
	if err!=nil{
		return
	}
	defer  file.Close()

	GerandoGraficosRegressaoLinearsTrainMachineLearning(file)
}

func GerandoGraficos (file *os.File)(error){
	var err error
	concreteDF:=dataframe.ReadCSV(file)

	log.Println(concreteDF)
	y_strenght :=concreteDF.Col("strength").Float()
	for _,colName := range concreteDF.Names(){
		pts := make(plotter.XYs,concreteDF.Nrow())
		for i,floatVal := range concreteDF.Col(colName).Float(){
			pts[i].X = floatVal
			pts[i].Y = y_strenght[i]
		}
		plot, err := plot.New()
		if err!=nil{
			log.Println(err)
			return err
		}
		plot.X.Label.Text = colName
		plot.Y.Label.Text="strengh"
		plot.Add(plotter.NewGrid())

		s,err:= plotter.NewScatter(pts)
		s.GlyphStyle.Color = color.RGBA{R:255,B:128,A:255}
		s.GlyphStyle.Radius = vg.Points(3)

		plot.Add(s)
		err = plot.Save(4*vg.Inch,4*vg.Inch,"scatter/"+colName+"_Scatter.png")

		if err!=nil{
			log.Println(err)
			return err
		}

	}
	return err
}

func TrainMachineLearning(file *os.File){

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 9

	trainingData,err:= reader.ReadAll()
	if err!=nil{
		log.Println(err)
		return
	}

	var r regression.Regression
	r.SetObserved("Strength")
	r.SetVar(0,"Cement")

	for i, record :=range trainingData{
		if i==0{
			continue
		}
		strengthVal,err:=strconv.ParseFloat(record[8],64)
		if err!=nil{
			log.Println(err)
		}
		cementVar,err:=strconv.ParseFloat(record[0],64)
		if err!=nil{
			log.Println(err)
		}
		r.Train(regression.DataPoint(strengthVal,[]float64{cementVar}))
	}
	r.Run()

	log.Println("Formula regressão:\n\n%v",r.Formula)

}

func TestMachineLearning(file *os.File){

	var media int
	media = 300

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 9

	dadosConcreto,err:= reader.ReadAll()
	if err!=nil{
		log.Println(err)
		return
	}

	var r regression.Regression
	r.SetObserved("Strength")
	r.SetVar(0,"Cement")

	for i, record :=range dadosConcreto[:media]{
		if i==0{
			continue
		}
		strengthVal,err:=strconv.ParseFloat(record[8],64)
		if err!=nil{
			log.Println(err)
		}
		cementVar,err:=strconv.ParseFloat(record[0],64)
		if err!=nil{
			log.Println(err)
		}
		r.Train(regression.DataPoint(strengthVal,[]float64{cementVar}))
	}
	r.Run()
	log.Println("Formula regressão:\n\n",r.Formula,"\n\n")

	var mAE float64

	for i,record :=range dadosConcreto[media:] {
		if i==0{
			continue
		}
		valStrength,err:= strconv.ParseFloat(record[8],64)
		if err!=nil{
			log.Println(err)
		}

		valCement, err:= strconv.ParseFloat(record[0],64)
		if err!=nil{
			log.Println(err)
		}

		ypredicted,err:=r.Predict([]float64{valCement})

		mAE += math.Abs(valStrength-ypredicted/float64(len(dadosConcreto[media:])))
	}

	log.Println("MAE =",mAE)
}


func GerandoGraficosRegressaoLinearsTrainMachineLearning(file *os.File){

	concreteDF := dataframe.ReadCSV(file)

	yVals:= concreteDF.Col("strength").Float()

	pts:= make(plotter.XYs, concreteDF.Nrow())

	ptsPred := make(plotter.XYs,concreteDF.Nrow())

	for i,floatVal:= range concreteDF.Col("cement").Float(){
		pts[i].X = floatVal
		pts[i].Y = yVals[i]
		ptsPred[i].X = floatVal
		ptsPred[i].Y = predict(floatVal)
	}
	p,_ := plot.New()
	p.X.Label.Text = "Cimento"
	p.Y.Label.Text = "strength"

	p.Add(plotter.NewGrid())

	s,_ := plotter.NewScatter(pts)
	s.GlyphStyle.Color = color.RGBA{R:255,B:128,A:255}
	s.GlyphStyle.Radius = vg.Points(3)

	l,_:= plotter.NewLine(ptsPred)
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Dashes = []vg.Length{vg.Points(5),vg.Points(5)}
	l.LineStyle.Color = color.RGBA{B:255,A:255}

	p.Add(s,l)
	p.Save(4*vg.Inch, 4*vg.Inch,"regression_line.png")

}
func predict(cement float64) float64{
	return 11.07+cement*0.1
}