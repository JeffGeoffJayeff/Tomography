# The basic process of this program is Take image -> Do Scans -> Use Scans to see what x we can get


from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import matplotlib.pyplot as plt

filename = "freddy.png"
shortfile = "hurhur"
InputFolder = "Input/"
OutputFolder = "Output/"
WorkingFolder = "Workingdir/"

ReconstructionWidth = 16 #Size of the reconstruction grid, just gonna do it as a square
Detectors = 21 #How many detectors are used
Rotations =  np.arange(0,181,1) #[0,45,90,135] # In Degrees, angles start from the positive x-axis



class Ray:
    def __init__(self,gridsize=100,x_1=-1,y_1=-1,x_2=-1,y_2=-1,length=1,): #Two modes to do this, either define length or define points
        self.coords = np.array([[x_1,x_1+length],[y_1,y_1]])
        self.theta = 0 #How much the ray has been rotated from starting position
        self.center = gridsize/2 #gridsize is the reconstruction size, the center ray should be half way in the grid
        self.length = length
        if x_2 != -1:
            self.x_2 = x_2
            self.CalculateLength()
        if y_2 != -1:
            self.y_2 = y_2
            self.CalculateLength()
        #Making the rotate command like this because it will make dealing with rotations easier later
    def RotateBy(self,theta): #Rotates the ray about the center of the reconstruction grid, which is irritatingly not at 0,0
        theta = math.radians(theta)
        rotation_array = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        self.coords = self.coords - self.center #Translation, tried +=, didn't work, this is moving the ray to be centered around the origin so the rotation matrix rotates it as expected
        self.coords = np.matmul(rotation_array,self.coords) #Rotation
        self.coords = self.coords + self.center #Translation
    def RotateTo(self,desiredtheta):
        deltatheta = desiredtheta - self.theta
        self.RotateBy(deltatheta)
        self.theta=desiredtheta
    def SetTheta(self,theta): 
        self.theta = theta
    def CalculateLength(self):
        deltax = abs(self.x_1-self.x_2)
        deltay = abs(self.y_1-self.y_2)
        self.length = math.sqrt(deltax**2+deltay**2)
    @property 
    def x_1(self):
        return round(self.coords[0,0],4)
    @x_1.setter
    def x_1(self,new):
        self.coords[0,0] = new
    @property
    def x_2(self):
        return round(self.coords[0,1],4)
    @x_2.setter
    def x_2(self,new):
        self.coords[0,1] = new
    @property
    def y_1(self):
        return round(self.coords[1,0],4)
    @y_1.setter
    def y_1(self,new):
        self.coords[1,0] = new
    @property
    def y_2(self):
        return round(self.coords[1,1],4)
    @y_2.setter
    def y_2(self,new):
        self.coords[1,1] = new
    @property
    def x_1_float(self):
        return self.coords[0,0]
    @property
    def x_2_float(self):
        return self.coords[0,1]
    @property
    def y_1_float(self):
        return self.coords[1,0]
    @property
    def y_2_float(self):
        return self.coords[1,1]
    def returncoords(self):
        return [self.x_1,self.y_1,self.x_2,self.y_2]

class AMatrix:
    def __init__(self):
        self.ReconstructionWidth = 3 #Reconstruction grid width
        self.Detectors = Detectors #How many detectors
        self.Rotations = Rotations # Storing the rotation angles
        self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
        self.Rays: list[Ray]= [] # To store the ray objects that make up the detector
        self.minx = 0 #These two are used for drawing the plot, may have other uses later, basically stores the min and max coordinates of the Rays before rotation
        self.maxx = 0
    #Setting Methods, should probably use the property thing but this works    
    def SetReconstruction(self,NewWidth):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,NewDetectors):
        if NewDetectors > 0:
            self.Detectors = NewDetectors
            print(f"Now using {self.Detectors} detectors")
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
        xCoordinate = (self.ReconstructionWidth-rayGroupWidth)/2+0.5#self.ReconstructionWidth/2*(1-math.sqrt(2))# #Same for all rays
        self.minx = xCoordinate
        self.maxx = xCoordinate+raySpacing*self.Detectors
        for i in range(0,self.Detectors):
            self.Rays.append(Ray(gridsize=self.ReconstructionWidth,x_1=xCoordinate,y_1=xCoordinate+raySpacing*i,length=rayGroupWidth))
        
    # Making and moving Rays
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
            i.RotateTo(theta)
    # Image management
    def ClearImage(self):
        TheDraw = ImageDraw.Draw(self.APicture)
        TheDraw.rectangle([0,0,self.ReconstructionWidth,self.ReconstructionWidth],fill="Black",outline="Black")
    def SaveImage(self,filename="DefaultA.bmp"):
        self.APicture.save(WorkingFolder+filename)
    # Make Matrices
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
    # Debugging
    def DebugCreateX(self): 
        self.Rays.append(Ray(x_1=self.minx,x_2=self.maxx,y_1=self.minx,y_2=self.maxx,gridsize=self.ReconstructionWidth))
        self.Rays.append(Ray(x_1=self.minx,x_2=self.maxx,y_1=self.maxx,y_2=self.minx,gridsize=self.ReconstructionWidth))
class xMatrix: #This should be done after the AMatrix is made, manages everything related to the constructed xMatrix
    def __init__(self,Amatrix,ReconWidth=3):
        self.ReconstructionWidth = ReconWidth
        self.xarray = np.zeros(self.ReconstructionWidth**2) #Initializes the array for the xmatrix
        self.Amatrix: AMatrix = Amatrix
        print("About to calculate C*At*R, stand by...")
        self.CAtransR = np.matmul(np.matmul(self.Amatrix.CMatrix,self.Amatrix.AMatrix_Transpose),self.Amatrix.RMatrix) #This saves alot of time 
        print("C*At*R calculated")
        self.Pmatrix = np.empty([1,1]) #Starts off empty, should be imported ASAP
        self.iteration = 0
        self.maxiteration = 0 
        self.frames = [] #Used to store frames to make a gif later
        self.detectors = 0
    #Setting Up Stuff Section
    def ImportPVector(self,p):
        self.Pmatrix = p #this is wasteful but its python so who cares
        print("P vector copied")
    def SetReconstruction(self,NewWidth=0): #Lets the reconstruction width be changed, don't know if this will be used
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
            self.xarray = np.zeros(self.ReconstructionWidth**2)
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,detectors):
        self.detectors = detectors
    def SetIterations(self,iter): #Set the number of iterations to do
        if iter >= 0:
            self.maxiteration = iter
        else:
            print("Use a positive number")
    #Iteration Section
    def Iterate(self): #Do a single iteration of the xarray 
        #self.xarray = self.xarray + np.matmul((np.matmul(np.matmul(self.Amatrix.CMatrix,self.Amatrix.AMatrix_Transpose),self.Amatrix.RMatrix)),(p-np.matmul(self.Amatrix.AMatrix,self.xarray)))
        self.xarray = self.xarray + np.matmul(self.CAtransR,(self.Pmatrix-np.matmul(self.Amatrix.AMatrix,self.xarray)))
        # As a future improvement calculate the CAtRP and the CAtRA and then reuse it instead of recalculating it everytime
        self.xarray = np.abs(self.xarray) #If you don't do this then you get random white dots because when going from signed to unsigned the low negative values turn to 255
    def DoAllIterations(self): #Does all of iterations 
        for i in range(0,self.maxiteration):
            print(f"Starting iteration {i} of {self.maxiteration}...",end='')
            self.Iterate()
            print(f"iteration {i} calculated, saving image...",end='')
            self.SaveImage(WorkingFolder+f"{shortfile}Iteration{i}.bmp")
            print(f"image saved! Iteration {i} completed!")
            MakeSinogram(np.matmul(self.Amatrix.AMatrix,self.xarray),WorkingFolder+f"{shortfile}SinogramIteration{i}.bmp",np.size(Rotations),self.detectors) #TODO: Disable for performance, just comment line out
        print(f"Saving final output image")
        self.SaveImage(OutputFolder+f"{shortfile}.bmp")
        print(f"Final image saved!")
    #Saving Stuff Section
    def SaveImage(self,filename): #Function to save the image and add it to frames to make a gif later
        output = np.reshape(self.xarray.astype(np.int8),(self.ReconstructionWidth,self.ReconstructionWidth))
        max = np.max(output)
        if max > 0:
            scale = 255//max
        else:
            scale = 1
        outputimage = Image.fromarray(output*scale,"L")
        self.frames.append(outputimage)
        outputimage.save(filename)
    def SaveGif(self,filename=OutputFolder+f"{shortfile}.gif"):
        self.frames[0].save(filename,
               save_all = True, append_images = self.frames[1:],
               optimize = False, duration = 50,loop = 0)
        print("GIF saved!")
        
def MakeSinogram(inputarray,filename,numberofrotations,detectors):
    max = np.max(inputarray)
    if max > 0:
        scale = 255/max
    else:
        scale = 1
    output = np.abs(np.reshape((inputarray*scale).astype(np.int8),(numberofrotations,detectors)))
    outputimage = Image.fromarray(output,"L")
    outputimage.save(filename)
    
def PlotAMatrix(AMatrix: AMatrix, ax: plt.axes):
    rays: list[Ray] = AMatrix.Rays
    for i in rays:
        ax.plot([i.x_1,i.x_2],[i.y_1,i.y_2])
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
    #thedog.SetOptimalDetectors()
    thedog.CreateRays()
    thedog.DebugCreateX()
    thedog.DrawRays()
    fig, ax = plt.subplots()
    for i in range(0,180):
        ax.imshow(img)
        PlotAMatrix(thedog,ax)
        thedog.RotateRays(i)
        ax.set_xlim([thedog.minx-5,thedog.maxx+5])
        ax.set_ylim([thedog.minx-5,thedog.maxx+5])  
        
        plt.savefig(f"Workingdir/Torture/{i}.png")
        ax.cla()  
    return
    thedog.CreateAMatrix() #These three functions should probably be linked into another function but whatever
    p = np.matmul(thedog.AMatrix,xarray)
    MakeSinogram(p,f"{OutputFolder}Sinograms/{shortfile}.bmp",np.size(Rotations),thedog.Detectors)
    thedog.CreateCMatrix()
    thedog.CreateRMatrix()
    #plt.bar(np.array(range(0,thedog.Detectors)),p[0:thedog.Detectors])
    #plt.show()
    thecat = xMatrix(thedog,thedog.ReconstructionWidth)
    thecat.ImportPVector(p)
    thecat.SetIterations(100)
    thecat.SetDetectors(thedog.Detectors)
    thecat.DoAllIterations()
    thecat.SaveGif(OutputFolder+f"{shortfile}.gif")
    
main()