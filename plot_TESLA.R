# 
# used in visualize the result of IDP_stepwise_training.py
#
library(data.table)
library(ggplot2)
library(dplyr)

extension = "eps"

n.add.ATP <- function(n,n.gamma){
  return(paste0(n," with ",n.gamma))
}

n.ref = "Original IDP"
n.RTESLA = "R-TESLA"
n.TESLA = "TESLA"
n.gamma = "ATP"
n.width = 6
n.height= 4

n.RTESLA.gamma = n.add.ATP(n.RTESLA,n.gamma)
n.TESLA.gamma  = n.add.ATP(n.TESLA,n.gamma)

naming <- read.csv("naming.csv",stringsAsFactors = F)

n.search <- function(n,...){
  dots <- list(...)
  if("from" %in% names(dots)) from = dots[["from"]] else from = "short"
  if("to" %in% names(dots)) to = dots[["to"]] else to = "long"
  if("naming" %in% names(dots)) naming = dots[["naming"]] else naming = naming
  res = naming[naming[[from]]==n,to]
  return(res)
}



setwd("C:/Users/mmnet/Desktop/IDP/output")
dirs <- grep(pattern="1019",list.dirs(getwd(),recursive = F),value = T)
f <- list()
for(this.dir in dirs){
  res_file <- grep("result",list.files(this.dir,pattern = ".csv",full.names = T),value = T)
  alg = ifelse(grepl("RTESLA",this.dir),n.RTESLA,n.TESLA)
  alg = ifelse(grepl("ATP",this.dir),n.add.ATP(alg,n.gamma),alg)
  for(i in seq_along(res_file)){
    tf <- fread(res_file[i])
    fn.parsed <- strsplit(basename(res_file[i]),split = "_")[[1]]
    tf$alg <- rep(alg,nrow(tf))
    tf$atp <- rep(as.numeric(fn.parsed[3],nrow(tf)))
    tf$func <- rep(fn.parsed[1],nrow(tf))
    tf$alpha <- rep(as.numeric(fn.parsed[2]),nrow(tf))

    f <- rbind(f,tf)
  }
}


f$IDP <- f$IDP * 100
f$accu <- f$accu * 100

ref <- list()
res_file <- list.files(paste0(getwd(),"/original_both/"),pattern="result",full.names = T)
for(i in seq_along(res_file)){
  tf <- fread(res_file[i])
  fn.parsed <- strsplit(basename(res_file[i]),split = "_")[[1]]
  tf$alg <- rep("IDP",nrow(tf))
  tf$func <- rep(fn.parsed[2],nrow(tf))
  ref <- rbind(ref,tf)
}
ref$accu <- ref$accu*100
ref$IDP <- ref$IDP*100

rm(tf,fn.parsed,alg,i,res_file,this.dir)

f[grep(n.gamma,alg),]$profile <- gsub("epochs", "rounds", f[grep(n.gamma,alg),]$profile)

f$profile <- gsub("at","at optimizing",f$profile)
f$profile <- gsub(" 1 rounds"," 1 round",f$profile)
f$profile <- gsub(" 1 rounds"," 1 round",f$profile)
f$profile <- gsub(" 1 epochs"," 1 epoch",f$profile)
f$profile <- gsub(" 1 epochs"," 1 epoch",f$profile)

f <- rbind(f,ref,fill=T)

f[f$alg=="IDP",]$alg <- "Original IDP"
# f[f$alg=="Profile",]$alg <- "Multiple Profile IDP"

bf <- f
tar <- c("alg","atp")
bf[,avg.accu:=sum(accu)/length(accu),by=.(profile)]
best <- bf[,.(at=profile[which.max(avg.accu)]),by=c(tar)]$at
bf <- as.data.frame(bf)
bf <- bf[(bf$profile %in% best),c("IDP","accu",tar)]

# bf[is.na(bf)] <- 0
# bf$color <- rep("IDP",nrow(bf))
# bf[grep("^TESLA",bf$alg),"color"] <- rep("TESLA",length(grep("^TESLA",bf$alg)))
# bf[grep("^R-TESLA",bf$alg),"color"] <- rep("R-TESLA",length(grep("^R-TESLA",bf$alg)))
# bf$color <- as.factor(bf$color)
# bf$line <- rep("No",nrow(bf))
# bf[grep("ATP$",bf$alg), "line"] <- rep("Yes",length(grep("ATP$",bf$alg)))
# bf$line <- as.factor(bf$line)
bf$alg <- as.factor(bf$alg)
ggplot(bf,aes(x=IDP,y=accu,
              colour=alg))+
  scale_y_continuous(breaks = seq(0,100,by=5))+
  scale_x_continuous(breaks = seq(10,100,by=10))+
  geom_line(size=1)+
  geom_point(size=2)+
    labs(colour=n.search("alg"))+
  ggtitle(paste0("Trainable Coefficients by ",n.gamma," on MLP (MNIST)"))+
  xlab(n.search("IDP",to="long.complete"))+
  ylab(n.search("accu",to="long.complete"))+
  scale_colour_manual(values=c("grey","#f16913","#8c2d04","#4292c6","#084594"))+
  theme_bw()+
  coord_cartesian(ylim = c(65, 100)) +
  theme(plot.title = element_text(hjust=0.5),
        legend.position = c(0.8,0.35),
        legend.text=element_text(size=12),
        legend.title = element_text(size=12))
ggsave(filename = paste0(getwd(),"/","TrainableCoefficient_by_alg_best.",extension),
       width = n.width,height = n.height,units="in")


#################################
# compare the best by condition #
#################################

myplot <- function(f, tar, main, comp, by){
  pref <- f[(f$alg==comp),c("IDP","accu","profile",tar),with=F]
  pref[,avg.accu:=sum(accu)/length(accu),by=.(profile)]
  best <- pref[,.(at=profile[which.max(avg.accu)]),by=c(tar[length(tar)])]$at
  pref <- as.data.frame(pref)
  pref <- pref[(pref$profile %in% best),c("IDP","accu",tar)]
  
  for(this in main){
    #print(this)
    pf <- f[f[[tar[1]]]==this,c("IDP","accu","profile",tar),with=F]
    pf[,avg.accu:=sum(accu)/length(accu),by=.(profile)]
    best <- pf[,.(at=profile[which.max(avg.accu)]),by=c(tar[length(tar)])]$at
    pf <- as.data.frame(pf)
    pf <- pf[(pf$profile %in% best),c("IDP","accu",tar)]
    print(best)
    pf <- rbind(pf,pref)
    pf <- setDT(pf)
    pf <- pf[,avg:=mean(accu),by=c(tar[1],tar[2])]
    summary <- pf[,.(get(tar[1]),avg,get(tar[2]))] %>% unique()
    print(summary)
    
    if(by==2){
      p1 <- ggplot(pf,aes(x=IDP,y=accu,
                          colour=as.factor(pf[[tar[2]]]),
                          lty=factor(pf[[tar[1]]],levels=c(this,comp))))+
        labs(lty=n.search(tar[1]), colour=n.search(tar[2]))+
        guides(colour = guide_legend(order=1),
               lty = guide_legend(order=2))
    }
    if(by==1){
      p1 <- ggplot(pf,aes(x=IDP,y=accu,
                          colour=factor(pf[[tar[1]]],levels=c(this,comp))))+
        labs(colour=n.search(tar[1]))
    }
  
  
  p1 <- p1+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1.2)+
    geom_point(size=2)+
    ggtitle(paste0(this," on MLP (MNIST)"))+
    xlab(n.search("IDP",to="long.complete"))+
    ylab(n.search("accu",to="long.complete"))+
    theme_bw()+
    coord_cartesian(ylim = c(50, 100)) +
    theme(plot.title = element_text(hjust=0.5),
          legend.position = c(0.8,0.35),
          legend.text=element_text(size=12),
          legend.title = element_text(size=12),
          legend.box.just = "left")
  #png(filename = paste0(getwd(),"/",this,"_by_",tar[length(tar)],"_best.png"),width = 6,height = 4,res = 200,units="in")
  #print(p1)
  #dev.off()
  ggsave(plot = p1, filename = paste0(getwd(),"/",this,"_by_",tar[length(tar)],"_best.",extension),
         width = n.width,height = n.height,units="in")
  }
}
# main, comp, by, fixed
tar = c("alg","func")
main = c( n.TESLA,n.RTESLA)
comp = n.ref
# pref <- f[f$alg==comp,c("IDP","accu",tar),with=F]

myplot(f,tar,main,comp,2)


tar = c("alg","atp")
main = c(n.TESLA.gamma,n.RTESLA.gamma)
comp = n.ref

myplot(f[func=="all-one"],tar,main,comp,1)

algs = c(n.TESLA,n.RTESLA)
funcs = c("all-one","harmonic","linear")
for(this in algs){
  for(thf in funcs){
    pf <- f[(alg==this)&(func==thf)]
    tt <- strsplit(split = "IDP = ",pf$profile) %>% do.call(what="rbind")
    i <- c(which(abs(diff(round(as.numeric(gsub(")","",tt[,2])),2)))>0),length(tt[,2]))-30
    if(i[length(i)]==length(tt[,2])){
      
    }
    # i <- which(!duplicated(tt[,2],fromLast = T))
    
    opt <- as.character(round(as.numeric(gsub(")","",tt[i,2])) * 100,0))
    for(ii in seq_along(opt)){
      # if(ii==1){opt[ii] <- paste0("1st Task: ", opt[ii],"% IDP")}
      # else if(ii==2){opt[ii] <- paste0("2nd Task: ", opt[ii],"% IDP")}
      # else if(ii==3){opt[ii] <- paste0("3rd Task: ", opt[ii],"% IDP")}
      # else {opt[ii] <- paste0(ii,"th Task: ", opt[ii],"% IDP")}
      opt[ii] <- paste0("Task ",ii," : ",opt[ii],"% DP")
    }
    # opt <- paste0("After optimized at ",round(opt,0),"% IDP")
    pf <- pf[profile %in% pf$profile[i]]
    
    p1 <- ggplot(pf,aes(x=IDP,y=accu,
                        colour=as.factor(profile)))+
      scale_y_continuous(breaks = seq(0,100,by=5))+
      scale_x_continuous(breaks = seq(10,100,by=10))+
      geom_line(size=1.2)+
      geom_point(size=2)+
      ggtitle(paste0(this,": Validation Accuracy Curve during Training"))+
      xlab(n.search("IDP",to="long.complete"))+
      ylab(n.search("accu",to="long.complete"))+
      scale_colour_manual(values=c('#66c2a5','#fc8d62','#8da0cb','#e78ac3','#ffd92f','#a6d854')[seq(1,length(opt))],
                          name=thf,
                          labels=opt)+
      # scale_colour_discrete(values=heat.colors(length(opt)),
      #                       name=thf,
      #                       labels=opt)+
      theme_bw()+
      coord_cartesian(ylim = c(40, 100)) +
      theme(plot.title = element_text(hjust=0.5),
            legend.position = c(0.75,0.35),
            legend.text=element_text(size=12),
            legend.title = element_text(size=12),
            legend.text.align = 0)
    ggsave(plot = p1, filename = paste0(getwd(),"/",this,"_",thf,"_process.",extension),
           width = n.width,height = n.height,units="in")
  }
}


### VGG 16 experiment: 100 50###

setwd("C:/Users/mmnet/Desktop/IDP/output/")

filename = "p100_50.csv"

f <- fread(paste0(getwd(),"/",filename))
opt = gsub("p","",filename)
opt = gsub("\\.csv","",opt)
opt = strsplit(split = "_",opt) %>% unlist() %>% as.numeric()

f$IDP <- (f$V1*5+5)
f$V1 <- NULL
library(reshape2)
f <- melt(f,id.vars = "IDP",value.name = "accu")
f$variable <- as.character(f$variable)
tmp <- strsplit(split="_",f$variable) %>% do.call(what="rbind")
colnames(tmp) <- c("alg","func")
f <- cbind(f,tmp)
f[f$alg=="IDP",]$alg <- "Original IDP"
f[f$alg=="Profile",]$alg <- "Multiple Profile IDP"
funcs = c("all-one","linear")
for(thf in funcs){
  pf <- f[(func==thf)]
  print(pf %>% group_by(alg) %>% summarize(avg=mean(accu[-c(1,2,3)])))
  p1 <- ggplot(pf,aes(x=IDP,y=accu,
                      colour=factor(alg,levels=c("Original IDP","Multiple Profile IDP","TESLA","R-TESLA"))))+
    scale_y_continuous(breaks = seq(0,100,by=10))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_vline(xintercept = opt,size=1.2,lty=3, colour="grey")+
    geom_line(size=1.2)+
    geom_point(size=2)+
    ggtitle(paste0(paste0(sort(opt),collapse = "/"), "% DP, ",stringr::str_to_title(thf)," Coefficients"))+
    xlab(n.search("IDP",to="long.complete"))+
    ylab(n.search("accu",to="long.complete"))+
    labs(colour=n.search("alg",to="long"))+
    
    # scale_colour_manual(values=c('#fdbe85','#fd8d3c','#e6550d','#a63603'), 
    #                     name=NULL,
    #                     labels=opt)+
    # scale_colour_discrete(name=thf,
    #                       labels=opt)+
    theme_bw()+
    coord_cartesian(ylim = c(0, 100)) +
    theme(plot.title = element_text(hjust=0.5,size=18),
          axis.text.x= element_text(size=14),
          axis.text.y= element_text(size=14),
          axis.title.x = element_text(size=14),
          axis.title.y = element_text(size=14),
          legend.position="none",
          legend.position = c(0.15,0.65),
          legend.text=element_text(size=12),
          legend.title = element_text(size=12),
          legend.text.align = 0
          )
  ggsave(plot = p1, filename = paste0(getwd(),"/","VGG16_",thf,"_best.",extension),
         width = n.width,height = n.height,units="in")
}

savehere <- f[variable=="IDP_linear"]

filename = "p100_50_20.csv"

f <- fread(paste0(getwd(),"/",filename))
opt = gsub("p","",filename)
opt = gsub("\\.csv","",opt)
opt = strsplit(split = "_",opt) %>% unlist() %>% as.numeric()

f$IDP <- (f$V1*5+5)
f$V1 <- NULL
library(reshape2)
f <- melt(f,id.vars = "IDP",value.name = "accu")
f$variable <- as.character(f$variable)
# f$variable <- gsub("_100_50_20","",f$variable)
tmp <- strsplit(split="_",f$variable) %>% do.call(what="rbind") %>% as.data.frame()
colnames(tmp) <- c("alg","func")
f <- cbind(f,tmp[,c(1,2)])
f[f$alg=="IDP",]$alg <- "Original IDP"
f[f$alg=="Profile",]$alg <- "Multiple Profile IDP"

funcs = c("linear")
for(thf in funcs){
  pf <- f[(func==thf)]
  pf <- rbind(pf,savehere)
  print(pf %>% group_by(alg) %>% summarize(avg=mean(accu[-c(1,2,3)])))
  p1 <- ggplot(pf,aes(x=IDP,y=accu,
                      colour=factor(alg,levels=c("Original IDP","Multiple Profile IDP","TESLA","R-TESLA"))))+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_vline(xintercept = opt,size=1.2,lty=3, colour="grey")+
    geom_line(size=1.2)+
    geom_point(size=2)+
    ggtitle(paste0(paste0(sort(opt),collapse = "/"), "% DP, ",stringr::str_to_title(thf)," Coefficients"))+
    xlab(n.search("IDP",to="long.complete"))+
    ylab(n.search("accu",to="long.complete"))+
    labs(colour=n.search("alg",to="long"))+
    
    # scale_colour_manual(values=c('#fdbe85','#fd8d3c','#e6550d','#a63603'), 
    #                     name=NULL,
    #                     labels=opt)+
    # scale_colour_discrete(name=thf,
    #                       labels=opt)+
    theme_bw()+
    coord_cartesian(ylim = c(0, 100)) +
    theme(plot.title = element_text(hjust=0.5,size=18),
          axis.text.x= element_text(size=14),
          axis.text.y= element_text(size=14),
          axis.title.x = element_text(size=14),
          axis.title.y = element_text(size=14),
          legend.position = c(0.8,0.35),
          legend.text=element_text(size=12),
          legend.title = element_text(size=12),
          legend.text.align = 0)
          
  ggsave(plot = p1, filename = paste0(getwd(),"/","VGG16_3p_",thf,"_best.",extension),
         width = n.width,height = n.height,units="in")
}



###  gamma ###

files <- list.files(getwd(),recursive=T)
files <- grep(pattern="all-one_50_1_r2.csv",files,value=T)
files <- grep(pattern="ATP_4point",files,value=T)
want <- c("88","38")

for(i in seq_along(files)){
  f <- fread(files[i])
  tt <- strsplit(split="_", colnames(f)) %>% do.call(what="rbind")
  r2 <- setDF(f)[,which(tt[,2]==want[i])]
  if(i==1){
    pf <- data.table(x=seq(1,length(r2)),y=r2)
    pf$alg <- rep(n.RTESLA.gamma,length(r2))
  }else{
    tf <- data.table(x=seq(1,length(r2)),y=r2)
    tf$alg <- rep(n.TESLA.gamma,length(r2))
    pf <- rbind(pf,tf)
  }
}
tf <- data.table(x=seq(1,length(r2)),y=1/seq(1,length(r2)))
tf$alg <- rep("Harmonic",length(r2))
pf <- rbind(pf,tf)


p1 <- ggplot(pf,aes(x=x,y=y,colour=factor(alg,levels=c("Harmonic",n.TESLA.gamma,n.RTESLA.gamma))))+
  geom_line(size=1)+
  scale_y_continuous(breaks = seq(0,1,by=0.1))+
  scale_x_continuous(breaks = seq(5,100,by=5))+
  geom_line(size=1.2)+
  #geom_point(size=2)+
  ggtitle(paste0("Trained Coefficients using ",n.gamma))+
  xlab("Index")+
  ylab("Value")+
  scale_colour_manual(values=c("grey","#8c2d04","#084594"),
                      name=n.search("alg"),
                      labels=c("Harmonic",n.TESLA.gamma,n.RTESLA.gamma))+
  # scale_colour_discrete(name=thf,
  #                       labels=opt)+
  theme_bw()+
  coord_cartesian(ylim = c(0, 1)) +
  theme(plot.title = element_text(hjust=0.5),
        legend.position = c(0.85,0.65),
        legend.text=element_text(size=12),
        legend.title = element_text(size=12),
        legend.text.align = 0)
ggsave(plot = p1, filename = paste0(getwd(),"/","ATP_MLP_gamma.",extension),
       width = n.width,height = n.height,units="in")


