setwd('C:/Blog/Faces')

#Read in required packages
if(!require(pixmap)) install.packages("pixmap", dependencies=T)
library(pixmap)
if(!require(h2o)) install.packages("h2o", dependencies=T)
library(h2o)

#Get  names of train image files
train.names<-list.files(path="train/face")

#For test image files get equal number of faces and not-faces
test.faces<-list.files(path="test/face")
test.notfaces<-sample(list.files(path="train/non-face"),
                      length(test.faces),
                      replace=F)
test.names<-append(test.faces, test.notfaces)

#Get pixel vectors for these files
train.vectors<-list()
test.vectors<-list()
for (f in train.names){
  x=read.pnm(file=paste('train/face/', f, sep=""))
  pixvec=as.vector(t(x@grey))
  pixvec=as.integer(pixvec>mean(pixvec))
  train.vectors[[f]]=pixvec
}
for (f in test.names[1:length(test.faces)]){
  x=read.pnm(file=paste('test/face/', f, sep=""))
  pixvec=as.vector(t(x@grey))
  pixvec=as.integer(pixvec>mean(pixvec))
  test.vectors[[f]]=pixvec
}
for (f in test.names[(length(test.faces)+1):length(test.names)]){
  x=read.pnm(file=paste('train/non-face/', f, sep=""))
  pixvec=as.vector(t(x@grey))
  pixvec=as.integer(pixvec>mean(pixvec))
  test.vectors[[f]]=pixvec
}

#Create dataframes of pixel vectors
train<-as.data.frame(t(as.data.frame(train.vectors)))
test<-as.data.frame(t(as.data.frame(test.vectors)))

#Initialize h2o cluster
h2o.init(nthreads=-1, max_mem_size='4g')
h2o.clusterInfo()

#Create h2o dataframes for train and test sets
train.hex<-as.h2o(train, destination_frame="train.hex")
test.hex<-as.h2o(test, destination_frame="test.hex")

#Create an autoencoder model
ae_model<-h2o.deeplearning(x=1:ncol(train.hex),
                           training_frame=train.hex,
                           activation="Tanh",
                           autoencoder=T,
                           hidden=c(400, 400),
                           distribution='AUTO', 
                           epochs=10, 
                           reproducible=T,
                           seed=1801,
                           ignore_const_cols=F,
                           export_weights_and_biases=T)

#Do reconstruction of test pixel vectors with the autoencoder
test.recon<-h2o.anomaly(ae_model, test.hex, per_feature=F)
test.recon<-as.data.frame(test.recon)
test.recon$filename=row.names(test)

#Sort by decreasing reconstruction error
test.recon<-test.recon[order(-test.recon$Reconstruction.MSE), ]

#Create a column for the truth value
test.recon$face=0
test.recon$face[grepl('^cmu', test.recon$filename)]=1

#Make the prediction based on a threshold
test.recon$predictedface=0
#Try an index for the threshold:
thresh=ceiling(nrow(test.recon)/2)
test.recon$predictedface[test.recon$Reconstruction.MSE<=test.recon$Reconstruction.MSE[thresh]]=1
#Create a confusion matrix for the given threshold
cm=table(test.recon$face, test.recon$predictedface)
print(cm)
#This is about an 82% accuracy for both classes
#Threshold can be changed based on how important it is to detect faces vs. detect not-faces

#Try to see what neurons are being triggered while making a prediction
layer1=as.data.frame(h2o.deepfeatures(ae_model, test.hex, layer=1))
row.names(layer1)=row.names(test)
layer2=as.data.frame(h2o.deepfeatures(ae_model, test.hex, layer=2))
row.names(layer2)=row.names(test)

#Get filenames of the most facey and non-facey images
top20notface=test.recon$filename[1:20]
top20face=test.recon$filename[(nrow(test.recon)-19):nrow(test.recon)]


#Create a rotate function for matrices (i.e., rotate clockwise by 90deg)
rotate <- function(x) t(apply(x, 2, rev))

#Plot these images
for (filename in top20notface){
    
  #Create a grid plot
  par(mfrow=c(1,4), oma=c(0,0,2,0))
  
  #Original image
  x=read.pnm(file=paste('te/', filename, sep=""))
  plot(x, main='Original Image')
  
  #High contrast image
  m=matrix(as.integer(test[filename,]), nrow=19, ncol=19, byrow=T)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(m), col=grey(c(0:1)), axes=F, xlim=c(0,1), ylim=c(0,1), asp=1, main='High Contrast Image')
  
  #Layers
  image(t(matrix(as.numeric(layer1[filename,]))), asp=1, axes=F, col=heat.colors(100), main='Layer 1')
  image(t(matrix(as.numeric(layer2[filename,]))), asp=1, axes=F, col=heat.colors(100), main='Layer 2')
  
  #Title
  if(test.recon$face[test.recon$filename==filename]==1)
    Actual='Face'
  else Actual='Not Face'
  if(test.recon$predictedface[test.recon$filename==filename]==1)
    Predicted='Face'
  else Predicted='Not Face'
  titlestring=sprintf("Actual - %s, Predicted - %s", Actual, Predicted)
  title(titlestring, outer=TRUE)
  
}
for (filename in top20face){
  
  #Create a grid plot
  par(mfrow=c(1,4), oma=c(0,0,2,0))
  
  #Original image
  x=read.pnm(file=paste('te/', filename, sep=""))
  plot(x, main='Original Image')
  
  #High contrast image
  m=matrix(as.integer(test[filename,]), nrow=19, ncol=19, byrow=T)
  image(rotate(m), col=grey(c(0:1)), axes=F, xlim=c(0,1), ylim=c(0,1), asp=1, main='High Contrast Image')
  
  #Layers
  image(t(matrix(as.numeric(layer1[filename,]))), asp=1, axes=F, col=heat.colors(100), main='Layer 1')
  image(t(matrix(as.numeric(layer2[filename,]))), asp=1, axes=F, col=heat.colors(100), main='Layer 2')
  
  #Title
  if(test.recon$face[test.recon$filename==filename]==1)
    Actual='Face'
  else Actual='Not Face'
  if(test.recon$predictedface[test.recon$filename==filename]==1)
    Predicted='Face'
  else Predicted='Not Face'
  titlestring=sprintf("Actual - %s, Predicted - %s", Actual, Predicted)
  title(titlestring, outer=TRUE)
  
}

#Visualize weights
#Weight matrix between input and layer 1
w.in.1=as.data.frame(h2o.weights(ae_model, matrix_id=1))
dev.off()
par(mfrow=c(20,20))
for (i in 1:nrow(w.in.1)){
  m=matrix(as.numeric(w.in.1[i,]), nrow=19, ncol=19, byrow=T)
  image(rotate(m), col=topo.colors(3), axes=F, asp=1, main=paste('At Neuron', i, sep=" "))
}

