# The basic process of this program is Take image -> Do Scans -> Use Scans to see what x we can get


from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import matplotlib.pyplot as plt

filename = "Sans2.png"
shortfile = "SansWhoever"
InputFolder = "Input/"
OutputFolder = "Output/"
WorkingFolder = "Workingdir/"

ReconstructionWidth = 64 #Size of the reconstruction grid, just gonna do it as a square
Detectors = 20 #How many detectors are used
Rotations = np.arange(0, 90,1) # In Degrees, angles start from the positive x-axis

class Ray:
    def __init__(self,x_1=-1,y_1=-1,length=1,gridsize=100):
        self.coords = np.array([[x_1,x_1+length],[y_1,y_1]])
        self.theta = 0 #Current angle of ray in degrees
        self.center = gridsize/2
        #Making the rotate command like this because it will make dealing with rotations easier later
    def rotate(self): #Rotates the ray about the center of the reconstruction grid, which is irritatingly not at 0,0
        theta = math.radians(self.theta)
        rotation_array = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        self.coords = self.coords - self.center #Translation, tried +=, didn't work
        self.coords = np.matmul(rotation_array,self.coords) #Rotation
        self.coords = self.coords + self.center #Translation
    def setTheta(self,theta): 
        self.theta = theta
    @property 
    def x_1(self):
        return round(self.coords[0,0],0)
    @property
    def x_2(self):
        return round(self.coords[0,1],0)
    @property
    def y_1(self):
        return round(self.coords[1,0],0)
    @property
    def y_2(self):
        return round(self.coords[1,1],0)
    def returncoords(self):
        return [self.x_1,self.y_1,self.x_2,self.y_2]

class AMatrix:
    def __init__(self):
        self.ReconstructionWidth = 3 #Reconstruction grid width
        self.Detectors = 5 #How many detectors
        self.Rotations = Rotations # Storing the rotation angles
        self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
        self.Rays = [] # To store the ray objects that make up the detector
        
    def SetReconstruction(self,NewWidth):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,NewDetectors):
        if NewDetectors > 0:
            self.Detectors = NewDetectors
        else:
            print("There must be more than 0 detectors")
    def SetOptimalDetectors(self):
        self.SetDetectors(math.ceil(self.ReconstructionWidth*math.sqrt(2)))
    def AppendRotation(self,AdditionalRotation):
        self.Rotations.append(AdditionalRotation)
    def AppendRotations(self,AdditionalRotations): #This is for adding multiple rotations
        self.Rotations.extend(AdditionalRotations)
    def CreateRays(self):
        rayGroupWidth = self.ReconstructionWidth*math.sqrt(2) #Multiplying by the square root two to get the diagonal width
        raySpacing = rayGroupWidth/self.Detectors
        xCoordinate = (self.ReconstructionWidth-rayGroupWidth)/2 #Same for all rays
        for i in range(0,self.Detectors):
            self.Rays.append(Ray(xCoordinate,xCoordinate+raySpacing*i,rayGroupWidth,self.ReconstructionWidth))
    def DrawRay(self,rayNum):
        drawer = ImageDraw.Draw(self.APicture)
        if rayNum < self.Detectors:
            coords = self.Rays[rayNum].returncoords()
            drawer.line(coords,fill="White",width=0)
        else:
            return #don't know what to do if condition fails
    def DrawRays(self):
        for i in range(0,len(self.Rays)):
            self.DrawRay(i)
    def RotateRays(self,theta):
        for i in self.Rays:
            i.setTheta(theta)
            i.rotate()
    def ClearImage(self):
        TheDraw = ImageDraw.Draw(self.APicture)
        TheDraw.rectangle([0,0,self.ReconstructionWidth,self.ReconstructionWidth],fill="Black",outline="Black")
    def SaveImage(self,filename="DefaultA.bmp"):
        self.APicture.save(WorkingFolder+filename)
    def CreateAMatrix(self): #Actually making the matrix :)
        self.AMatrix = np.zeros([self.Detectors*len(self.Rotations),(self.ReconstructionWidth)**2])
        self.CreateRays()
        for angleNum in range(0,len(self.Rotations)): #This is basically having PIL draw the lines that each ray will make, and then adding it as a row vector to its spot in the A matrix
            self.RotateRays(self.Rotations[angleNum])
            for ray in range(0,self.Detectors):
                self.ClearImage()
                self.DrawRay(ray)
                self.AMatrix[angleNum*self.Detectors+ray] = np.asarray(self.APicture).reshape(-1)
        print(f"A Matrix created, with a size of {np.shape(self.AMatrix)}")
        self.AMatrix_Transpose = self.AMatrix.transpose() #Precomputing this
        print("Transpose of A Matrix calculated")
    def CreateCMatrix(self):
        columns = np.shape(self.AMatrix)[1]
        self.CMatrix = np.zeros([columns,columns]) #This needs to be a square
        for j in range(0,columns):
            denominator = np.sum(self.AMatrix[:,j])#Summing specific column
            self.CMatrix[j,j] = 1/denominator
        print("C Matrix created")
    def CreateRMatrix(self):
        rows = np.shape(self.AMatrix)[0]
        self.RMatrix = np.zeros([rows,rows]) #This needs to be a square
        for i in range(0,rows):
            denominator = np.sum(self.AMatrix[i,:])
            if denominator > 0:
                self.RMatrix[i,i] = 1/denominator
            else:
                self.RMatrix[i,i] = 0
        print("R Matrix created")
class xMatrix: #This should be done after the AMatrix is made, manages everything related to the constructed xMatrix
    def __init__(self,Amatrix,ReconWidth=3):
        self.ReconstructionWidth = ReconWidth
        self.xarray = np.zeros(self.ReconstructionWidth**2)
        self.Amatrix = Amatrix
        print("About to calculate C*At*R, stand by...")
        self.CAtransR = np.matmul(np.matmul(self.Amatrix.CMatrix,self.Amatrix.AMatrix_Transpose),self.Amatrix.RMatrix)
        print("C*At*R calculated")
        self.Pmatrix = np.empty([1,1])
        self.iteration = 0
        self.maxiteration = 0
        self.frames = []
    def ImportPVector(self,p):
        self.Pmatrix = p #this is wasteful but its python so who cares
        print("P vector copied")
    def SetReconstruction(self,NewWidth=0):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
            self.xarray = np.zeros(self.ReconstructionWidth**2)
        else:
            print("Reconstruction grid must be larger than 0")
    def SetIterations(self,iter):
        if iter >= 0:
            self.maxiteration = iter
        else:
            print("Use a positive number")
    def Iterate(self):
        #self.xarray = self.xarray + np.matmul((np.matmul(np.matmul(self.Amatrix.CMatrix,self.Amatrix.AMatrix_Transpose),self.Amatrix.RMatrix)),(p-np.matmul(self.Amatrix.AMatrix,self.xarray)))
        self.xarray = self.xarray + np.matmul(self.CAtransR,(self.Pmatrix-np.matmul(self.Amatrix.AMatrix,self.xarray)))
        # As a future improvement calculate the CAtRP and the CAtRA and then reuse it instead of recalculating it everytime
        self.xarray = np.abs(self.xarray) #If you don't do this then you get random white dots because when going from signed to unsigned the low negative values turn to 255
    def DoAllIterations(self):
        for i in range(0,self.maxiteration):
            print(f"Starting iteration {i}...",end='')
            self.Iterate()
            print(f"iteration {i} complete!")
    def SaveImage(self,filename):
        output = np.reshape(self.xarray.astype(np.int8),(self.ReconstructionWidth,self.ReconstructionWidth))
        max = np.max(output)
        if max > 0:
            scale = 255//max
        else:
            scale = 1
        outputimage = Image.fromarray(output*scale,"L")
def main():
    try:
        path = "Input/"+filename
        print(f"Loading {path}...")
        img = Image.open(path).resize([ReconstructionWidth,ReconstructionWidth]).save(WorkingFolder+"AsBMP.bmp")
        img = Image.open(WorkingFolder+"AsBMP.bmp")
        print(f"{path} loaded!")
    except IOError:
        print("File issues!!")
        pass
    xarray = np.asarray(img).reshape(-1) # Making x
    print(f"x array made with shape {np.shape(xarray)}")
    thedog = AMatrix()
    thedog.SetReconstruction(ReconstructionWidth)
    thedog.SetOptimalDetectors()
    thedog.CreateRays()
    thedog.DrawRays()
    thedog.CreateAMatrix()
    p = np.matmul(thedog.AMatrix,xarray)
    thedog.CreateCMatrix()
    thedog.CreateRMatrix()
    #plt.bar(np.array(range(0,thedog.Detectors)),p[0:thedog.Detectors])
    #plt.show()
    thecat = xMatrix(thedog,thedog.ReconstructionWidth)
    thecat.ImportPVector(p)
    frames = []
    for i in range(0,100):
        thecat.Iterate()
        print(f"Iteration:{i}")
        output = np.reshape(thecat.xarray.astype(np.int8),(ReconstructionWidth,ReconstructionWidth))
        max = np.max(output)
        if max > 0:
            scale = 255//max
        else:
            scale = 1
        outputimage = Image.fromarray(output*scale,"L")
        frames.append(outputimage)
        outputimage.save(WorkingFolder+f"{shortfile}Iteration{i}.bmp")
    outputimage.save(OutputFolder+f"{shortfile}.bmp")
    frames[0].save(OutputFolder+f"{shortfile}.gif",
               save_all = True, append_images = frames[1:],
               optimize = False, duration = 50,loop = 0)
    
main()