# 
# used in visualize the result of IDP_stepwise_training.py
#

library(data.table)
library(ggplot2)
library(dplyr)
library(animation)
setwd("~/IDP")
####
## iterative training
####
this.dir <- "output/1018_RTESLA_ATP/"
res_file <- grep("result",list.files(this.dir,full.names = T),value = T)
log_file <- grep("log",list.files(this.dir,full.names = T),value=T)
byATP= ifelse(grep("ATP",this.dir),TRUE,FALSE)
Randomized = ifelse(grep("RTESLA",this.dir),TRUE,FALSE)
profile_type <- c("harmonic","all-one")
for(i in seq_along(res_file)){
    if(i==1){
        f <- fread(res_file[i])
        fn.parsed <- strsplit(basename(res_file[i]),split = "_")[[1]]
        f$alpha <- rep(as.numeric(fn.parsed[2]),nrow(f))
        f$alternate <- rep(as.numeric(fn.parsed[3],nrow(f)))
    }
    else{
        tf <- fread(res_file[i])
        fn.parsed <- strsplit(basename(res_file[i]),split = "_")[[1]]
        tf$alpha <- rep(as.numeric(fn.parsed[2]),nrow(tf))
        tf$alternate <- rep(as.numeric(fn.parsed[3],nrow(tf)))
        f <- rbind(f,tf)
    }
}
rm(tf)
for(i in seq(1,9)){
    f$profile <- gsub(paste0(" ",i," "),paste0(" 0",i," "),f$profile)
    #f$profile <- gsub(paste0(" ",i," "),paste0(" 0",i," "),f$profile)
}
tmp <- do.call("rbind",strsplit(gsub("\\)","",f$profile),split="="))
tmp[,2] <- round(as.numeric(tmp[,2]),2)
f$profile <- paste0(tmp[,1],"= ",tmp[,2],")")
f$IDP.train <- tmp[,2]
rm(tmp)
idx <- f$alternate==1
if(Randomized){
    f[idx]$profile <- paste0("Randomized TESLA+ATP, ",f[idx]$profile)
    f[!idx]$profile <- paste0("Randomized TESLA, ",f[!idx]$profile)
}else{
    f[idx]$profile <- paste0("TESLA+ATP, ",f[idx]$profile)
    f[!idx]$profile <- paste0("TESLA, ",f[!idx]$profile)
}


rm(idx)
f$IDP <- f$IDP*100
f$accu <- f$accu*100
fg <- f[grep("gamma",profile)]
fw <- f[grep("W,b",profile)]
fg$profile <- gsub("\\(train gamma in the "," \\(",fg$profile)
fw$profile <- gsub("\\(train W,b in the "," \\(",fw$profile)
fg$profile <- gsub("epochs","rounds",fg$profile)
fw$profile <- gsub("epochs","rounds",fw$profile)
fg$profile <- gsub("at","at optimizing",fg$profile)
fw$profile <- gsub("at","at optimizing",fw$profile)
fg$profile <- gsub("\\(1 rounds","\\(1 round",fg$profile)
fw$profile <- gsub("\\(1 rounds","\\(1 round",fw$profile)
if(!byATP){
    fg = fw
}
fg =f 
types <- unique(do.call(what="rbind",strsplit(basename(res_file),split = "_"))[,2])
types <- as.numeric(types)
fg <- fg[order(alpha)]
types <- unique(fg$alpha)

for(prof in profile_type){
    for(type in types){
        for(b in c(0,1)){
            pf <- fg[(alpha==type)&(alternate==b)&grepl(prof,profile)]
            # myorder <- stringi::stri_extract_all(levels(as.factor(pf$profile)), regex = "[0-9]+") %>% unlist() %>% as.numeric() %>% order(na.last=F)
            # mylevel<- levels(as.factor(pf$profile))[myorder]
            # pf$profile <- factor(pf$profile,levels = mylevel)
            pf$profile <- gsub(", TESLA+ATP","",pf$profile)
            pf$profile <- gsub(", TESLA","",pf$profile)
            pf$profile <- as.factor(pf$profile)
            if(b==0){
                mytitle <- paste0("MLP (MNIST), TESLA,  alpha = ",type/100)
            }else{
                mytitle <- paste0("MLP (MNIST), TESLA+ATP, alpha = ",type/100)
            }
            ggplot(pf,aes(x=IDP,y=accu,colour=profile))+
                scale_y_continuous(breaks = seq(0,100,by=5))+
                scale_x_continuous(breaks = seq(10,100,by=10))+
                geom_line(size=1)+
                geom_point(size=2)+
                ggtitle(mytitle)+
                xlab("IDP (%)")+
                ylab("Classification accuracy (%)")+
                theme_bw()+
                coord_cartesian(ylim = c(0, 100)) +
                theme(plot.title = element_text(hjust=0.5),
                      legend.position = c(0.5,0.3),
                      legend.text=element_text(size=10),
                      legend.title = element_text(size=15))
            ggsave(paste0(this.dir,"idp_",prof,"_alpha",type,"_b",b,".png"),width = 10,height = 8,units="in")
        }
    }
}

pf <- lapply(types,function(type){
    res = data.table()
    for(prof in profile_type){
        for(b in c(0,1)){
            pf <- fg[(alpha==type)&(alternate==b)&grepl(prof,profile)]
            tt <- pf[,(sum=sum(accu)),by=profile]
            nn <- tt[which.max(tt$V1),(profile)]
            print(paste0(prof,", alternate = ",b,", alpha = ",type/100,", average = ",round(max(tt$V1))/19))
            #print(paste0(type," improved ",round((max(tt$V1)-tt[grep("\\(original",profile)]$V1)/19,3),"%"))
            if(nrow(res)==0){
                res <- pf[profile==nn]
            }else{
                res <- rbind(pf[profile==nn],res)
            }
        }
    }
    return(res)
})
pf <- do.call("rbind",pf)
pf$profile <- paste0("alpha = ",pf$alpha/100,", ",pf$profile)
ori <- fread("output/original_both/ori_harmonic_result.csv")
ori$accu <- ori$accu*100
ori$IDP <- ori$IDP*100
ori$profile <- gsub("\\(fixed gamma ep20\\)"," (original)",ori$profile)
pf <- rbind(pf[,.(IDP,accu,profile)],ori)
# add original
# pf <- rbind(pf,fg[grep("ori",profile)])
# pf$type = paste0("Loss at IDP=",pf$loss.IDP,"%, ",pf$profile)
# mylevel = unique(pf$type)
# pf$type = factor(pf$type,levels = mylevel)
color7 <- c("#fd8d3c",
            "#6baed6",
            "#e6550d",
            "#3182bd",
            "#a63603",
            "#08519c",
            "#969696")
color4_ATP <-  c("#fd8d3c","#e6550d","#a63603","#969696")
color4 <- c("#fd8d3c","#e6550d","#a63603","#969696")

if(length(unique(pf$profile))>5) mycolor = color7 else mycolor = color4_ATP

ggplot(pf,aes(x=IDP,y=accu,colour=profile))+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1)+
    geom_point(size=2)+
    ggtitle(paste0("MLP (MNIST), comparison among the best ones"))+
    xlab("IDP (%)")+
    ylab("Classification accuracy (%)")+
    scale_color_manual(values=mycolor)+
    theme_bw()+
    coord_cartesian(ylim = c(50, 100)) +
    theme(plot.title = element_text(hjust=0.5),
          legend.position = c(0.5,0.3),
          legend.text=element_text(size=10),
          legend.title = element_text(size=15))
ggsave(paste0(this.dir,"idp_all_best.png"),width = 10,height = 8,units="in")

ppf <- pf[grepl("TESLA\\+ATP,", pf$profile)|grepl("original",pf$profile)]
if(length(unique(ppf$profile))>5) mycolor = color7 else mycolor = color4_ATP
ggplot(ppf,aes(x=IDP,y=accu,colour=profile))+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1)+
    geom_point(size=2)+
    ggtitle(paste0("MLP (MNIST), comparison among the best ones"))+
    xlab("IDP (%)")+
    ylab("Classification accuracy (%)")+
    scale_color_manual(values=mycolor)+
    theme_bw()+
    coord_cartesian(ylim = c(50, 100)) +
    theme(plot.title = element_text(hjust=0.5),
          legend.position = c(0.5,0.3),
          legend.text=element_text(size=10),
          legend.title = element_text(size=15))
ggsave(paste0(this.dir,"TESLA_ATP_idp_all_best.png"),width = 10,height = 8,units="in")

ppf <- pf[grepl("TESLA\\,", pf$profile)|grepl("original",pf$profile)]
if(length(unique(ppf$profile))>5) mycolor = color7 else mycolor = color4
ggplot(ppf,aes(x=IDP,y=accu,colour=profile))+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1)+
    geom_point(size=2)+
    ggtitle(paste0("MLP (MNIST), comparison among the best ones"))+
    xlab("IDP (%)")+
    ylab("Classification accuracy (%)")+
    scale_color_manual(values=mycolor)+
    theme_bw()+
    coord_cartesian(ylim = c(50, 100)) +
    theme(plot.title = element_text(hjust=0.5),
          legend.position = c(0.5,0.3),
          legend.text=element_text(size=10),
          legend.title = element_text(size=15))
ggsave(paste0(this.dir,"TESLA_idp_all_best.png"),width = 10,height = 8,units="in")

r2_file <- grep("r2",list.files("output/",full.names = T),value = T)
r2_file <- grep("\\.csv",r2_file,value=T)
#### plot r2 according to original order ####
for(prof in profile_type){
    for(type in types){
        for(b in c(1)){
            tmp_r2_file <- r2_file[grep(paste0(prof,"_",type,"_",b),r2_file)]
            for(i in seq_along(tmp_r2_file)){
                f <- fread(tmp_r2_file[i])
                f$order <- seq(1,nrow(f))
                pf <- melt(f,id.vars = "order",variable.factor = FALSE)
            }
            tmp <- do.call("rbind",strsplit(pf$variable,split="_"))
            idx <- which(as.numeric(tmp[,2])<10)
            pf$variable[idx] <- paste0(tmp[idx,1],"_0",tmp[idx,2])
            pf$variable <- as.factor(pf$variable)
            
            if(b==0){
                mytitle <- paste0("Trainable Channel Coefficents,  alpha = ",type/100, ", TESLA")
            }else{
                mytitle <- paste0("Trainable Channel Coefficents,  alpha = ",type/100, ", TESLA+ATP")
            }
            ggplot(pf,aes(x=order,y=value,colour=as.factor(variable)))+
                scale_x_continuous(breaks = seq(10,100,by=10))+
                geom_line()+
                ggtitle(mytitle)+
                xlab("Channel Coefficient Index")+
                ylab("Channel Coefficient (gamma)")+
                theme_bw()+
                theme(plot.title = element_text(hjust=0.5),
                      #legend.position = "bottom",
                      legend.text=element_text(size=12),
                      legend.title = element_text(size=12))
            
            ggsave(paste0(this.dir,"result_r2_",type,"_b",b,".png"),width = 10,height = 8,units="in")
        }
    }
}

