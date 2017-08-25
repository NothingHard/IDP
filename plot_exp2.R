library(data.table)
library(ggplot2)
library(dplyr)
library(animation)
setwd("~/IDP")
####
## iterative training
####
res_file <- grep("result",list.files("output/",full.names = T),value = T)
log_file <- grep("log",list.files("output/",full.names = T),value=T)

for(i in seq_along(res_file)){
    if(i==1){
        f <- fread(res_file[i])
    }
    else{
        tf <- fread(res_file[i])
        f <- rbind(f,tf)
    }
}
rm(tf)
f$IDP <- f$IDP*100
f$accu <- f$accu*100
fg <- f[grep("gamma",profile)]
fw <- f[grep("W,b",profile)]
fg$profile <- gsub("\\(train gamma in the "," \\(",fg$profile)
fw$profile <- gsub("\\(train W,b in the "," \\(",fw$profile)
fg$profile <- gsub("epochs\\)","rounds\\)",fg$profile)
fw$profile <- gsub("epochs\\)","rounds\\)",fw$profile)
fg$profile <- gsub("\\(0 rounds\\)","\\(0 round\\)",fg$profile)
fw$profile <- gsub("\\(0 rounds\\)","\\(0 round\\)",fw$profile)
fg$profile <- gsub("\\(fixed gamma ep20\\)"," \\(original\\)",fg$profile)

types <- unique(do.call(what="rbind",strsplit(basename(res_file),split = "_"))[,1])
for(type in types){
    pf <- fg[grep(type,profile)]
    myorder <- stringi::stri_extract_all(levels(as.factor(pf$profile)), regex = "[0-9]+") %>% unlist() %>% as.numeric() %>% order(na.last=F)
    mylevel<- levels(as.factor(pf$profile))[myorder]
    pf$profile <- factor(pf$profile,levels = mylevel)
    # for(lvl in levels(pf$profile)){
    #     tf <- pf[profile==lvl]
    #     ggplot(tf,aes(x=IDP,y=accu))+
    #         scale_y_continuous(breaks = seq(0,100,by=5))+
    #         scale_x_continuous(breaks = seq(10,100,by=10))+
    #         geom_line(size=1)+
    #         geom_point(size=2)+
    #         ggtitle(paste0("MLP (MNIST), ", lvl))+
    #         xlab("IDP (%)")+
    #         ylab("Classification accuracy (%)")+
    #         theme_bw()+
    #         coord_cartesian(ylim = c(0, 100)) +
    #         theme(plot.title = element_text(hjust=0.5),
    #               legend.text=element_text(size=12),
    #               legend.title = element_text(size=12))
    #     ggsave(paste0("output/",type,"_idp_",lvl,".png"))
    # }
    
    ggplot(pf,aes(x=IDP,y=accu,colour=profile))+
        scale_y_continuous(breaks = seq(0,100,by=5))+
        scale_x_continuous(breaks = seq(10,100,by=10))+
        geom_line(size=1)+
        geom_point(size=2)+
        ggtitle(paste0("MLP (MNIST), ",type))+
        xlab("IDP (%)")+
        ylab("Classification accuracy (%)")+
        theme_bw()+
        coord_cartesian(ylim = c(0, 100)) +
        theme(plot.title = element_text(hjust=0.5),
              #legend.position = c(0.7,0.5),
              legend.text=element_text(size=13),
              legend.title = element_text(size=15))
    ggsave(paste0("output/result_idp_w_",type,".png"))
}

pf <- lapply(types,function(type){
    pf <- fg[grep(type, profile)]
    tt <- pf[,(sum=sum(accu)),by=profile]
    nn <- tt[which.max(tt$V1),(profile)]
    print(paste0(type," average ",round(max(tt$V1))/19))
    print(paste0(type," improved ",round((max(tt$V1)-tt[grep("\\(original",profile)]$V1)/19,3),"%"))
    return(fg[profile==nn])
})
pf <- do.call("rbind",pf)
# add original
pf <- rbind(pf,fg[grep("ori",profile)])

ggplot(pf,aes(x=IDP,y=accu,colour=profile))+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1)+
    geom_point(size=2)+
    ggtitle(paste0("MLP (MNIST), comparison among the best ones"))+
    xlab("IDP (%)")+
    ylab("Classification accuracy (%)")+
    theme_bw()+
    coord_cartesian(ylim = c(30, 100)) +
    theme(plot.title = element_text(hjust=0.5),legend.position = c(0.8,0.2))
ggsave(paste0("output/result_idp_all_best.png"))


r2_file <- grep("r2",list.files("output/",full.names = T),value = T)
r2_file <- grep("\\.csv",r2_file,value=T)
#### plot r2 according to original order ####
for(type in types){
    tmp_r2_file <- r2_file[grep(type,r2_file)]
    for(i in seq_along(tmp_r2_file)){
        f <- fread(tmp_r2_file[i])
        f$order <- seq(1,nrow(f))
        pf <- melt(f,id.vars = "order")
    }
    
    # for(lvl in levels(pf$variable)){
    #     tf <- pf[variable==lvl]
    #     ggplot(tf,aes(x=order,y=value))+
    #         geom_line()+
    #         ggtitle(paste0("Trainable Channel Coefficients, ", lvl))+
    #         xlab("Channel Coefficient Index")+
    #         ylab("Channel Coefficient (gamma)")+
    #         theme_bw()+
    #         theme(plot.title = element_text(hjust=0.5),
    #               legend.text=element_text(size=12),
    #               legend.title = element_text(size=12))+
    #         coord_cartesian(ylim=c(min(pf$value),max(pf$value)))
    #     ggsave(paste0("output/",type,"_r2_",lvl,".png"))
    # }
    
    ggplot(pf,aes(x=order,y=value,colour=as.factor(variable)))+
        geom_line()+
        ggtitle(paste0("Trainable Channel Coefficients, ",type))+
        xlab("Channel Coefficient Index")+
        ylab("Channel Coefficient (gamma)")+
        theme_bw()+
        theme(plot.title = element_text(hjust=0.5),
              #legend.position = "bottom",
              legend.text=element_text(size=12),
              legend.title = element_text(size=12))

    ggsave(paste0("output/result_r2_",type,".png"))
}

