# The basic process of this program is Take image -> Do Scans -> Use Scans to see what x we can get


from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import matplotlib.pyplot as plt

filename = "Steve2.png"
shortfile = "SteveParallel"
InputFolder = "Input/"
OutputFolder = "Output/"
WorkingFolder = "Workingdir/"
plotmatlab = False

ReconstructionWidth = 32 #Size of the reconstruction grid, just gonna do it as a square
Detectors = 21 #How many detectors are used
Rotations = [0,22.5,45,67.5,90,112.5,135,157.5] #np.arange(0,181,1) # # In Degrees, angles start from the positive x-axis



class Ray:
    def __init__(self,x_1:float=-1,y_1:float=-1,x_2:float=None,y_2:float=None,length:float=0,rotationOrigin:list=[0,0]): #Two modes to do this, either define length or define points
        self.coords = np.array([[0,0],[0,0]])
        if length != 0:
            self.coords = np.array([[x_1,x_1+length],[y_1,y_1]]) #This layout is probably bad practice but whatever
        else:
            self.coords = np.array([[x_1,x_2],[y_1,y_2]])
        if x_2 != None:
            self.x_2 = x_2
            self.CalculateLength()
        if y_2 != None:
            self.y_2 = y_2
            self.CalculateLength()
        self.theta = 0 #How much the ray has been rotated from starting position
        self.length = length
        self.rotationOrigin = np.array([[rotationOrigin[0]],[rotationOrigin[1]]])
            
        #Making the rotate command like this because it will make dealing with rotations easier later
    def RotateBy(self,theta:float=0): #Rotates the ray about the center of the reconstruction grid, which is irritatingly not at 0,0
        theta = math.radians(theta)
        rotation_array = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        self.coords = self.coords - self.rotationOrigin #Translation, tried +=, didn't work, this is moving the ray to be centered around the origin so the rotation matrix rotates it as expected
        self.coords = np.matmul(rotation_array,self.coords) #Rotation
        self.coords = self.coords + self.rotationOrigin #Translation
    def RotateTo(self,desiredtheta:float=0):
        deltatheta = desiredtheta - self.theta
        self.RotateBy(deltatheta)
        self.theta=desiredtheta
    def SetTheta(self,theta:float=0): 
        self.theta = theta
    def CalculateLength(self):
        deltax = abs(self.x_1-self.x_2)
        deltay = abs(self.y_1-self.y_2)
        self.length = math.sqrt(deltax**2+deltay**2)
        
    @property 
    def x_1(self):
        return self.coords[0,0]
    @x_1.setter
    def x_1(self,new:float):
        self.coords[0,0] = new
        
    @property
    def x_2(self):
        return self.coords[0,1]
    @x_2.setter
    def x_2(self,new:float):
        self.coords[0,1] = new
        
    @property
    def y_1(self):
        return self.coords[1,0]
    @y_1.setter
    def y_1(self,new:float):
        self.coords[1,0] = new
        
    @property
    def y_2(self):
        return self.coords[1,1]
    @y_2.setter
    def y_2(self,new:float):
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
        
    @property
    def rotationOrigin_x(self):
        return self.rotationOrigin[0][0]
    @rotationOrigin_x.setter
    def rotationOrigin_x(self,new:float):
        self.rotationOrigin[0][0] = new
    
    @property
    def rotationOrigin_y(self):
        return self.rotationOrigin[1][0]
    @rotationOrigin_y.setter
    def rotationOrigin_y(self,new:float):
        self.rotationOrigin[1][0] = new
    
class AMatrix:
    def __init__(self):
        self.ReconstructionWidth:int = 3 #Reconstruction grid width
        self.Detectors:int = Detectors #How many detectors 
        self.Rotations:list[float] = Rotations # Storing the rotation angles
        self.APicture:Image = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth)) #Stores the picture that is used to make a row of A
        self.Rays: list[Ray]= [] # To store the ray objects that make up the detector
        self.minx:float = 0 #These two are used for drawing the plot, may have other uses later, basically stores the min and max coordinates of the Rays before rotation
        self.maxx:float = 0
        self.center:list[float] = [self.ReconstructionWidth/2-0.5,self.ReconstructionWidth/2-0.5] #This -0.5 is to align with the center of the pixel rather than the bottom left of a pixel
        self.boundingBoxPoints = np.array([[0,0],[0,self.ReconstructionWidth],[self.ReconstructionWidth,self.ReconstructionWidth],[self.ReconstructionWidth,0]]) #The points that represent the four corners of the reconstruction grid, going clockwise from the bottom-left
    #Setting Methods, should probably use the property thing but this works    
    def SetReconstruction(self,NewWidth):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
            self.center = [self.ReconstructionWidth/2-.5,self.ReconstructionWidth/2-0.5]
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,NewDetectors):
        if NewDetectors > 0:
            self.Detectors = NewDetectors
            print(f"Now using {self.Detectors} detectors")
        else:
            print("There must be more than 0 detectors")
    def UpdateBoundingBox(self,newWidth):
        self.boundingBoxPoints = np.array([[0,0],[0,newWidth],[newWidth,newWidth],[newWidth,0]])
        self.boundingBoxPoints = self.boundingBoxPoints - self.ReconstructionWidth/2 #Translating the bounding box so its centered about the origin
        self.boundingBoxPoints = self.boundingBoxPoints*math.sqrt(2) #Scaling the bounding box about the origin by square root 2
        self.boundingBoxPoints = self.boundingBoxPoints + self.ReconstructionWidth/2 - 0.5 #-0.5 is to align the rays with the center of pixels better
    def SetOptimalDetectors(self):
        self.SetDetectors(math.ceil(self.ReconstructionWidth*math.sqrt(2)))
    def AppendRotation(self,AdditionalRotation):
        self.Rotations.append(AdditionalRotation)
    def AppendRotations(self,AdditionalRotations): #This is for adding multiple rotations
        self.Rotations.extend(AdditionalRotations)
    # Ray creation methods
    def CreateParallelRays(self): #Makes the rays parallel
        self.UpdateBoundingBox(self.ReconstructionWidth)
        rayGroupWidth = self.ReconstructionWidth*math.sqrt(2) #Multiplying by the square root two to get the diagonal width
        raySpacing = (self.boundingBox_TL[1]-self.boundingBox_BL[1])/(self.Detectors-1)#Lin space forumala
        xCoordinate = self.boundingBox_BL[0]#(self.ReconstructionWidth-rayGroupWidth)/2+0.5#self.ReconstructionWidth/2*(1-math.sqrt(2))# #Same for all rays
        self.minx = xCoordinate
        self.maxx = xCoordinate+raySpacing*(self.Detectors)
        for i in range(0,self.Detectors):
            #debug statement print(f"i: {i} | xCoodrinate: {xCoordinate} | yCoordinate: {xCoordinate + raySpacing*i}")
            self.Rays.append(Ray(x_1=xCoordinate,
                                 y_1=xCoordinate+raySpacing*i,
                                 length=rayGroupWidth,
                                 rotationOrigin=self.center))
    def CreateFanRays(self): #Makes the rays into a fan pattern
        self.UpdateBoundingBox(self.ReconstructionWidth)
        raySpacing = (self.boundingBox_TL[1]-self.boundingBox_BL[1])/(self.Detectors-1)
        fanningPointY = self.center[1] # Y coordinate of the points where the rays fan from
        fanningPointX = self.boundingBox_BR[0]
        fanWallx = self.boundingBox_BL[0]
        for i in range(0,self.Detectors):
            ycoord = self.boundingBox_BL[1]+i*raySpacing
            self.Rays.append(Ray(x_1=fanningPointX,
                                 y_1=fanningPointY,
                                 x_2=fanWallx,
                                 y_2=ycoord,
                                 rotationOrigin=self.center))
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
    def RotateRaysTo(self,theta):
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
        self.CreateParallelRays()
        for angleNum in range(0,len(self.Rotations)): #This is basically having PIL draw the lines that each ray will make, and then adding it as a row vector to its spot in the A matrix
            self.RotateRaysTo(self.Rotations[angleNum])
            for ray in range(0,self.Detectors):
                self.ClearImage()
                self.DrawRay(ray)
                # Debug statement: self.APicture.save(f"Workingdir/AMatrixDrawings/Angle{angleNum:10d}Ray{ray:10d}.bmp")
                self.AMatrix[angleNum*self.Detectors+ray] = np.asarray(self.APicture).reshape(-1)
        print(f"A Matrix created, with a size of {np.shape(self.AMatrix)}")
        self.AMatrix_Transpose = self.AMatrix.transpose() #Precomputing this
        print("Transpose of A Matrix calculated")
    def CreateCMatrix(self):
        columns = np.shape(self.AMatrix)[1]
        self.CMatrix = np.zeros([columns,columns]) #This needs to be a square
        for j in range(0,columns):
            denominator = np.sum(self.AMatrix[:,j])#Summing specific column
            if denominator > 0:
                self.CMatrix[j,j] = 1/denominator
            else:
                self.CMatrix[j,j] = 0
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
        self.Rays.append(Ray(x_1=self.boundingBox_BL[0],x_2=self.boundingBox_TR[0],y_1=self.boundingBox_BL[1],y_2=self.boundingBox_TR[1],rotationOrigin=self.center))
        self.Rays.append(Ray(x_1=self.boundingBox_TL[0],x_2=self.boundingBox_BR[0],y_1=self.boundingBox_TL[1],y_2=self.boundingBox_BR[1],rotationOrigin=self.center))
    # Properties
    @property
    def boundingBox_BL(self): #Returns X-Y coordinates in list format for the bounding box points, this is bottom left
        return self.boundingBoxPoints[0]
    @property
    def boundingBox_TL(self): #Top left
        return self.boundingBoxPoints[1]
    @property
    def boundingBox_TR(self): #Top right
        return self.boundingBoxPoints[2]
    @property
    def boundingBox_BR(self): #Botom Right
        return self.boundingBoxPoints[3]
class xMatrix: #This should be done after the AMatrix is made, manages everything related to the constructed xMatrix
    def __init__(self,Amatrix:AMatrix,ReconWidth:float=3):
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
            print(f"Starting iteration {i+1} of {self.maxiteration}...",end='')
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
        print(self.xarray)
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
        img2 = Image.open(path)
        print(f"{path} loaded!")
    except IOError:
        print("File issues!!")
        pass
    xarray = np.asarray(img).reshape(-1) # Making x
    print(f"x array made with shape {np.shape(xarray)}")
    thedog = AMatrix()
    thedog.SetReconstruction(ReconstructionWidth)
    thedog.SetOptimalDetectors()
    thedog.CreateParallelRays()
    #thedog.CreateFanRays()
    #thedog.DebugCreateX()
    thedog.DrawRays()
    fig, ax = plt.subplots()
    if plotmatlab: #Change at top to enable or disable
        for i in range(0,181):#Debug block
            ax.imshow(img)
            PlotAMatrix(thedog,ax)
            thedog.RotateRaysTo(i)
            ax.set_xlim([thedog.boundingBox_BL[0]-5,thedog.boundingBox_BR[0]+5])
            ax.set_ylim([thedog.boundingBox_BL[1]-5,thedog.boundingBox_TL[1]+5])  
            
            plt.savefig(f"Workingdir/Torture/{i}.png")
            ax.cla()   
        thedog.RotateRaysTo(0)
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