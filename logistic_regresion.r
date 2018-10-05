> library(readr)
> resp_sin <- read_delim("IRSI_2/UVG/Respuestas_cuestionarios_si_sin_psico.csv", 
+     ";", escape_double = FALSE, col_names = FALSE, 
+     col_types = cols(X1 = col_skip()), trim_ws = TRUE, 
+     skip = 2)
> View(resp_sin)
> class(resp_sin)
[1] "tbl_df"     "tbl"        "data.frame"
> summary(resp_sin)
       X2            X3             X4              X5             X6             X7             X8             X9            X10            X11            X12            X13     
 Min.   :0.0   Min.   :0.00   Min.   :18.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.00   Min.   :0.0  
 1st Qu.:0.0   1st Qu.:0.00   1st Qu.:19.00   1st Qu.:0.00   1st Qu.:0.00   1st Qu.:0.00   1st Qu.:0.00   1st Qu.:0.00   1st Qu.:2.00   1st Qu.:0.00   1st Qu.:0.00   1st Qu.:0.0  
 Median :0.5   Median :1.00   Median :21.00   Median :0.00   Median :0.00   Median :0.00   Median :0.00   Median :0.00   Median :2.00   Median :0.00   Median :1.00   Median :1.0  
 Mean   :0.5   Mean   :0.54   Mean   :21.14   Mean   :0.14   Mean   :1.68   Mean   :0.12   Mean   :0.04   Mean   :0.04   Mean   :1.94   Mean   :0.04   Mean   :0.74   Mean   :0.6  
 3rd Qu.:1.0   3rd Qu.:1.00   3rd Qu.:21.00   3rd Qu.:0.00   3rd Qu.:4.00   3rd Qu.:0.00   3rd Qu.:0.00   3rd Qu.:0.00   3rd Qu.:2.00   3rd Qu.:0.00   3rd Qu.:1.00   3rd Qu.:1.0  
 Max.   :1.0   Max.   :1.00   Max.   :38.00   Max.   :1.00   Max.   :5.00   Max.   :3.00   Max.   :1.00   Max.   :1.00   Max.   :2.00   Max.   :1.00   Max.   :1.00   Max.   :1.0  
                                                                                                                                                                                   
      X14            X15              X16              X17              X18        
 Min.   :0.00   Min.   : 72357   Min.   : 41964   Min.   :-19529   Min.   : 39179  
 1st Qu.:0.00   1st Qu.: 83952   1st Qu.: 55154   1st Qu.:  3689   1st Qu.: 65286  
 Median :0.00   Median : 98493   Median : 61772   Median : 10642   Median : 74354  
 Mean   :0.62   Mean   : 98487   Mean   : 69315   Mean   : 14010   Mean   : 84046  
 3rd Qu.:1.00   3rd Qu.:105336   3rd Qu.: 72605   3rd Qu.: 23736   3rd Qu.: 89436  
 Max.   :3.00   Max.   :154986   Max.   :153971   Max.   : 50571   Max.   :209407  
                NA's   :76       NA's   :76       NA's   :76       NA's   :76      
> exploration.model <- subset(resp_sin)
> exploration.model$X2 <- factor(exploration.model$X2)
> table(exploration.model$X2)

 0  1 
50 50 
> modelo.logit <- glm(X2 ~ X3+X4+X5+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18,data=exploration.model,family="binomial")
> summary(modelo.logit)

Call:
glm(formula = X2 ~ X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + 
    X12 + X13 + X14 + X15 + X16 + X17 + X18, family = "binomial", 
    data = exploration.model)

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.02632  -0.56028   0.03518   0.62690   1.67293  

Coefficients: (5 not defined because of singularities)
              Estimate Std. Error z value Pr(>|z|)  
(Intercept)  3.703e+01  2.983e+01   1.241   0.2145  
X3           7.937e-02  1.654e+00   0.048   0.9617  
X4          -1.405e+00  1.014e+00  -1.385   0.1660  
X5           8.656e-01  2.717e+00   0.319   0.7500  
X6           8.336e-01  7.365e-01   1.132   0.2577  
X7                  NA         NA      NA       NA  
X8                  NA         NA      NA       NA  
X9                  NA         NA      NA       NA  
X10                 NA         NA      NA       NA  
X11                 NA         NA      NA       NA  
X12         -6.070e+00  8.465e+00  -0.717   0.4733  
X13         -1.098e+00  3.545e+00  -0.310   0.7567  
X14         -9.369e-02  2.357e+00  -0.040   0.9683  
X15          1.274e-04  9.787e-05   1.302   0.1930  
X16         -3.097e-04  1.783e-04  -1.737   0.0824 .
X17         -1.369e-04  9.516e-05  -1.439   0.1501  
X18          6.928e-05  7.427e-05   0.933   0.3509  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 33.271  on 23  degrees of freedom
Residual deviance: 17.574  on 12  degrees of freedom
  (76 observations deleted due to missingness)
AIC: 41.574

Number of Fisher Scoring iterations: 7