# The basic process of this program is Take image -> Do Scans -> Use Scans to see what x we can get


from PIL import Image
from PIL import ImageDraw
import numpy as np
import math

filename = "BasicTest.png"
InputFolder = "Input/"
OutputFolder = "Output/"
WorkingFolder = "Workingdir/"

ReconstructionWidth = 16 #Size of the reconstruction grid, just gonna do it as a square
Detectors = 20 #How many detectors are used
Rotations = [0] # In Degrees, angles start from the positive x-axis


class AMatrix:
    def __init__(self):
        self.ReconstructionWidth = 3
        self.Detectors = 5
        self.Rotations = [0]
        self.APicture = Image.new("1",(ReconstructionWidth,ReconstructionWidth))
    def SetReconstruction(self,NewWidth):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,NewDetectors):
        if NewDetectors > 0:
            self.Detectors = NewDetectors
        else:
            print("There must be more than 0 detectors")
    def AppendRotation(self,AdditionalRotation):
        self.Rotations.append(AdditionalRotation)
    def AppendRotations(self,AdditionalRotations): #This is for adding multiple rotations
        self.Rotations.extend(AdditionalRotations)

class Ray:
    def __init__(self,x_1=-1,y_1=-1,length=1,gridsize=100):
        self.coords = np.array([[x_1,x_1+length],[y_1,y_1]])
        self.theta = 0 #Current angle of ray in degrees
        self.center = gridsize/2
    def rotate(self,theta): #Rotates the ray about the center of the reconstruction grid, which is irritatingly not at 0,0
        self.theta += theta
        theta = math.radians(theta)
        rotation_array = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        print(self.returncoords())
        self.coords = self.coords - self.center
        print(self.returncoords())
        self.coords = np.matmul(rotation_array,self.coords)
        print(self.returncoords())
        self.coords = self.coords + self.center
        print(self.returncoords())
    @property 
    def x_1(self):
        return self.coords[0,0]
    @property
    def x_2(self):
        return self.coords[0,1]
    @property
    def y_1(self):
        return self.coords[1,0]
    @property
    def y_2(self):
        return self.coords[1,1]
    def returncoords(self):
        return [self.x_1,self.y_1,self.x_2,self.y_2]
        

def main():
    try:
        path = "Input/"+filename
        img = Image.open(path).save(WorkingFolder+"AsBMP.bmp")
        img = Image.open(WorkingFolder+"AsBMP.bmp")
        draw = ImageDraw.Draw(img)
        draw.line([-128,-1,256,256],fill="white",width=0)
        draw.line([0,0,256,128],fill="white",width=0)
        img.save(WorkingFolder+"testing.bmp")
        img2 = Image.new("1",(101,101))
        
    except IOError:
        print("File issues")
        pass
    xarray = np.asarray(img).reshape(-1) # Making x
    thedog = AMatrix()
    print(thedog.ReconstructionWidth)
    thedog.SetReconstruction(10)
    print(thedog.ReconstructionWidth)
    thedog.APicture.save(WorkingFolder+"testingA.bmp")
    testray = Ray(44,50,9,101)
    draw2 = ImageDraw.Draw(img2)
    draw2.line(testray.returncoords(),fill="White")
    img2.save(WorkingFolder+"BeforeRot.bmp")
    testray.rotate(90)
    draw2.line(testray.returncoords(),fill="White")
    testray.rotate(45)
    draw2.line(testray.returncoords(),fill="White")
    img2.save(WorkingFolder+"AfterRot.bmp")
    
main()