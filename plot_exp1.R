library(data.table)
library(ggplot2)

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
f <- f[grep("gamma",profile)]

f$profile <- gsub("\\(train gamma in the last "," \\(",f$profile)
f$profile <- gsub("\\)"," on gamma\\)",f$profile)
f$profile <- gsub("\\(0 epochs on gamma\\)","\\(standard\\)",f$profile)
types <- unique(do.call(what="rbind",strsplit(basename(res_file),split = "_"))[,1])
for(type in types){
    pf <- f[grep(type,profile)]
    myorder <- stringi::stri_extract_all(levels(as.factor(pf$profile)), regex = "[0-9]+") %>% unlist() %>% as.numeric() %>% order(na.last=F)
    mylevel<- levels(as.factor(pf$profile))[myorder]
    pf$profile <- factor(pf$profile,levels = mylevel)
    ggplot(pf,aes(x=IDP,y=accu,colour=profile))+
        scale_y_continuous(breaks = seq(0,100,by=5))+
        scale_x_continuous(breaks = seq(10,100,by=10))+
        geom_line(size=1)+
        geom_point(size=2)+
        ggtitle(paste0("MLP (MNIST), ",type,", total # epochs = 100"))+
        xlab("IDP (%)")+
        ylab("Classification accuracy (%)")+
        theme_bw()+
        coord_cartesian(ylim = c(0, 100)) +
        theme(plot.title = element_text(hjust=0.5),
              legend.position = c(0.7,0.5),
              legend.text=element_text(size=13),
              legend.title = element_text(size=15))
    ggsave(paste0("output/result_idp_",type,".png"))
}

pf <- lapply(types,function(type){
    pf <- f[grep(type, profile)]
    tt <- pf[,(sum=sum(accu)),by=profile]
    nn <- tt[which.max(tt$V1),(profile)]
    print(paste0(type," improved ",round((max(tt$V1)-tt[grep("\\(0 epochs",profile)]$V1)/19,3),"%"))
    return(f[profile==nn])
})
pf <- do.call("rbind",pf)

ggplot(pf,aes(x=IDP,y=accu,colour=profile))+
    scale_y_continuous(breaks = seq(0,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1)+
    geom_point(size=2)+
    ggtitle(paste0("MLP (MNIST), comparison among the best ones, total # epochs = 100"))+
    xlab("IDP (%)")+
    ylab("Classification accuracy (%)")+
    theme_bw()+
    coord_cartesian(ylim = c(30, 100)) +
    theme(plot.title = element_text(hjust=0.5),legend.position = c(0.8,0.2))
ggsave(paste0("output/result_idp_all_best.png"))

#### plot log ####
for(type in types){
    tmp_log_file <- log_file[grep(type,log_file)]
    for(i in seq_along(tmp_log_file)){
        epoch_on_gamma <- as.numeric(strsplit(basename(tmp_log_file[i]),split="_")[[1]][2])
        epoch_on_gamma <- 100 - epoch_on_gamma
        if(i==1){
            f <- fread(tmp_log_file[i])
            f$epoch <- seq(1,nrow(f))
            f$epoch_on_gamma <- rep(epoch_on_gamma,nrow(f))
        }
        else{
            tf <- fread(tmp_log_file[i])
            tf$epoch <- seq(1,nrow(tf))
            tf$epoch_on_gamma <- rep(epoch_on_gamma,nrow(tf))
            f <- rbind(f,tf)
        }
    }
    pf <- melt(f[,.(epoch,epoch_on_gamma,val_accu)],
               id.vars = c("epoch","epoch_on_gamma"),
               variable.name = "type")
    pf$epoch_on_gamma <- as.character(pf$epoch_on_gamma)
    pf[epoch_on_gamma=="0"]$epoch_on_gamma <- rep("standard",sum(pf$epoch_on_gamma=="0"))
    pf$epoch_on_gamma <- as.factor(pf$epoch_on_gamma)
    myorder <- stringi::stri_extract_all(levels(as.factor(pf$epoch_on_gamma)), regex = "[0-9]+") %>% unlist() %>% as.numeric() %>% order(na.last=F)
    levels(pf$epoch_on_gamma) <- levels(pf$epoch_on_gamma)[myorder]
    
    
    #vlines <- data.frame(xint=c(90,50,10,100),
    #                     col=as.factor(sort(unique(pf$epoch_on_gamma))))
    ggplot(pf,aes(x=epoch,y=value,colour=epoch_on_gamma))+
        geom_line()+
        geom_point()+
        #geom_vline(data=vlines,
        #           aes(xintercept = xint,colour=col),linetype="dashed")+
        ggtitle(paste0("Epoch vs. Validation Accuracy, ",type))+
        xlab("Epoch")+
        ylab("Accuracy")+
        theme_bw()+
        theme(plot.title = element_text(hjust=0.5),
              legend.position = "bottom",
              legend.text=element_text(size=12),
              legend.title = element_text(size=12))
    ggsave(paste0("output/result_epoch_on_gamma_",type,".png"))
}

r2_file <-  grep("r2",list.files("output/",full.names = T),value = T)
#### plot r2 according to original order ####
for(type in types){
    tmp_r2_file <- r2_file[grep(type,r2_file)]
    for(i in seq_along(tmp_r2_file)){
        epoch_on_gamma <- as.numeric(strsplit(basename(tmp_r2_file[i]),split="_")[[1]][2])
        epoch_on_gamma <- 100 - epoch_on_gamma
        if(i==1){
            f <- fread(tmp_r2_file[i])
            f$order <- seq(1,nrow(f))
            f$epoch_on_gamma <- rep(epoch_on_gamma,nrow(f))
            
            ### print
            cor <- cor.test(f$before,f$after)
            print(paste(type,epoch_on_gamma,"> correltation =", round(cor$estimate,3),
                        ", p-value =",round(cor$p.value,3),
                        ", rmse=",sqrt(mean((f$before-f$after)^2)),
                        ", sd_ori=",sd(f$before),
                        ", sd_aft=",sd(f$after),sep = " "))
        }
        else{
            tf <- fread(tmp_r2_file[i])
            tf$order <- seq(1,nrow(tf))
            tf$epoch_on_gamma <- rep(epoch_on_gamma,nrow(tf))
            
            ### print
            cor <- cor.test(tf$before,tf$after)
            print(paste(type,epoch_on_gamma,"> correltation =", round(cor$estimate,3),
                        ", p-value =",round(cor$p.value,3),
                        ", rmse=",sqrt(mean((tf$before-tf$after)^2)),
                        ", sd_ori=",sd(tf$before),
                        ", sd_aft=",sd(tf$after),sep = " "))
            f <- rbind(f,tf)
        }
    }
    pf <- melt(f[,.(order,epoch_on_gamma,after,before)],
               id.vars = c("order","epoch_on_gamma"),
               variable.name = "type")
    pf$epoch_on_gamma <- as.character(pf$epoch_on_gamma)
    pf[epoch_on_gamma=="0"]$epoch_on_gamma <- rep("standard",sum(pf$epoch_on_gamma=="0"))
    pf$epoch_on_gamma <- as.factor(pf$epoch_on_gamma)
    
    myorder <- stringi::stri_extract_all(levels(as.factor(pf$epoch_on_gamma)), regex = "[0-9]+") %>% unlist() %>% as.numeric() %>% order(na.last=F)
    levels(pf$epoch_on_gamma) <- levels(pf$epoch_on_gamma)[myorder]
    
    ggplot(pf[type=="after"],aes(x=order,y=value,colour=epoch_on_gamma))+
        geom_line()+
        geom_point()+
        ggtitle(paste0("Trainable Channel Coefficients, ",type))+
        xlab("Channel Coefficient Index")+
        ylab("Channel Coefficient (gamma)")+
        theme_bw()+
        theme(plot.title = element_text(hjust=0.5),
              legend.position = "bottom",
              legend.text=element_text(size=12),
              legend.title = element_text(size=12))
    ggsave(paste0("output/result_r2_",type,".png"))
}

#### plot r2 after re-order ####
for(type in types){
    tmp_r2_file <- r2_file[grep(type,r2_file)]
    for(i in seq_along(tmp_r2_file)){
        epoch_on_gamma <- as.numeric(strsplit(basename(tmp_r2_file[i]),split="_")[[1]][2])
        epoch_on_gamma <- 100 - epoch_on_gamma
        if(i==1){
            f <- fread(tmp_r2_file[i])
            f <- f[order(-after)]
            f$order <- seq(1,nrow(f))
            f$epoch_on_gamma <- rep(epoch_on_gamma,nrow(f))
            
            ### print
            cor <- cor.test(f$before,f$after)
            print(paste(type,epoch_on_gamma,"> correltation =", round(cor$estimate,3),
                        ", p-value =",round(cor$p.value,3),
                        ", rmse=",sqrt(mean((f$before-f$after)^2)),
                        ", sd_ori=",sd(f$before),
                        ", sd_aft=",sd(f$after),sep = " "))
        }
        else{
            tf <- fread(tmp_r2_file[i])
            tf <- tf[order(-after)]
            tf$order <- seq(1,nrow(tf))
            tf$epoch_on_gamma <- rep(epoch_on_gamma,nrow(tf))
            
            ### print
            cor <- cor.test(tf$before,tf$after)
            print(paste(type,epoch_on_gamma,"> correltation =", round(cor$estimate,3),
                        ", p-value =",round(cor$p.value,3),
                        ", rmse=",sqrt(mean((tf$before-tf$after)^2)),
                        ", sd_ori=",sd(tf$before),
                        ", sd_aft=",sd(tf$after),sep = " "))
            f <- rbind(f,tf)
        }
    }
    pf <- melt(f[,.(order,epoch_on_gamma,after,before)],
               id.vars = c("order","epoch_on_gamma"),
               variable.name = "type")
    pf$epoch_on_gamma <- as.character(pf$epoch_on_gamma)
    pf[epoch_on_gamma=="0"]$epoch_on_gamma <- rep("standard",sum(pf$epoch_on_gamma=="0"))
    pf$epoch_on_gamma <- as.factor(pf$epoch_on_gamma)
    
    myorder <- stringi::stri_extract_all(levels(as.factor(pf$epoch_on_gamma)), regex = "[0-9]+") %>% unlist() %>% as.numeric() %>% order(na.last=F)
    levels(pf$epoch_on_gamma) <- levels(pf$epoch_on_gamma)[myorder]
    
    ggplot(pf[type=="after"],aes(x=order,y=value,colour=epoch_on_gamma))+
        geom_line()+
        geom_point()+
        ggtitle(paste0("Trainable Channel Coefficients, ",type))+
        xlab("Sort Coefficient in Descending Order")+
        ylab("Channel Coefficient (gamma)")+
        theme_bw()+
        theme(plot.title = element_text(hjust=0.5),
              legend.position = "bottom",
              legend.text=element_text(size=12),
              legend.title = element_text(size=12))
    ggsave(paste0("output/result_r2_reorder_",type,".png"))
}

# #### plot log ####
# for(i in seq_along(log_file)){
#     f <- fread(log_file[i])
#     f$epoch <- seq(1,nrow(f))
#     pf <- melt(f[,.(epoch,accu,val_accu)],
#                id.vars = "epoch",
#                variable.name = "type")
#     ggplot(pf,aes(x=epoch,y=value,colour=type))+
#         geom_line()+
#         geom_vline(xintercept = which.min(f$gamma_trainable==0),linetype="dotted")+
#         geom_point()+
#         ggtitle("Training log (accuracy)")+
#         xlab("Epoch")+
#         ylab("Accuracy")+
#         theme_bw()+
#         theme(plot.title = element_text(hjust=0.5))
#     ggsave(gsub("\\.csv","_accu\\.png",log_file[i]))
#     
#     pf <- melt(f[,.(epoch,loss,val_loss)],
#                id.vars = "epoch",
#                variable.name = "type")
#     ggplot(pf,aes(x=epoch,y=value,colour=type))+
#         geom_line()+
#         geom_vline(xintercept = which.min(f$gamma_trainable==0),linetype="dotted")+
#         ggtitle("Training log (loss)")+
#         xlab("Epoch")+
#         ylab("Loss (CE)")+
#         theme_bw()+
#         theme(plot.title = element_text(hjust=0.5))
#     ggsave(gsub("\\.csv","_loss\\.png",log_file[i]))
# }
