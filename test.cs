if(strcmp(c_call_flg,KRA_DATA)== 0)
{
    MEMSET(sql_ikkd_name);
MEMSET(sql_ikkd_cor_add1);
MEMSET(sql_ikkd_cor_add2);
MEMSET(sql_ikkd_cor_add3);
MEMSET(sql_ikkd_cor_city);
MEMSET(sql_ikkd_cor_ctry);
MEMSET(sql_ikkd_cor_pincd);
MEMSET(sql_ikkd_cor_state);
    MEMSET(vc_address_source);
    EXEC SQL
      SELECT NVL(MAR_ADDRESS_USED_FROM,'KRA'),
              NVL(MAR_KRA_ELIGIBLE ,'N'),
              NVL(MAR_FORM_NO,'*') /Ver 3.9/
      INTO :vc_address_source,
              :c_kra_eligible,
              :sql_mar_form_no /Ver 3.9/      
      FROM MAR_MBL_ACCOPN_RQST
      WHERE MAR_PAN_NO = :sql_mar_pan_no
      AND MAR_ACCOPN_SRC = 'E'
      AND MAR_STATUS = 'Z';
    if(SQLCODE != 0)
    {
      tpfree((char )ptr_fml_Obuffer);
errlog(c_ServiceName, "S31430",SQLMSG,(char )DEF_USR,DEF_SSSN, c_err_msg);
Fadd32(ptr_fml_Ibuffer, FML_ERR_MSG, c_err_msg, 0) ;
tpreturn(TPFAIL, 0, (char *)ptr_fml_Ibuffer, 0L, 0) ;  
    }

    userlog("sql_mar_pan_no :%s:,vc_address_source :%s:, c_kra_eligible :%c:",sql_mar_pan_no.arr,vc_address_source.arr,c_kra_eligible);   
    /* Ver 3.9 starts here /
SETNULL(sql_mar_form_no);
userlog("sql_mar_form_no |%s|",sql_mar_form_no.arr);
if(strcmp(sql_mar_form_no.arr,"") != '')
{
EXEC SQL
SELECT NVL(MAS_KRA_STTS,'N')
INTO :c_stage_stts
FROM MAS_MBL_APPL_STTS
WHERE MAS_FORM_NO = :sql_mar_form_no
AND MAS_FORM_STTS = 'A';
if(SQLCODE != 0 && SQLCODE != NO_DATA_FOUND)
{
errlog(c_ServiceName, "S31435", SQLMSG, DEF_USR, DEF_SSSN, c_err_msg);
Fadd32(ptr_fml_Ibuffer, FML_ERR_MSG, c_err_msg, 0);
tpreturn(TPFAIL , 0L , (char )ptr_fml_Ibuffer , 0L ,0);
}
userlog("c_stage_stts |%c|",c_stage_stts);
}
if(c_stage_stts 'Y')
{
strcpy(vc_stage_stts.arr,"Y");
}
else
{
strcpy(vc_stage_stts.arr,"N");
}
SETLEN(vc_stage_stts);
userlog("vc_stage_stts |%s|",vc_stage_stts.arr);
/ Ver 3.9 ends here _/
    if(c_kra_eligible 'Y' && strcmp(vc_address_source.arr,"KRA") == 0)
    {
      EXEC SQL
  SELECT IKKD_NAME,
                 IKKD_COR_ADD1,
                 IKKD_COR_ADD2,
                 IKKD_COR_ADD3,
                 NVL(ICM_CITY_DESC,IKKD_COR_CITY),
                 NVL(ICM_COUNTRY_DESC,IKKD_COR_CTRY),
                 IKKD_COR_PINCD,
                 NVL(ISM_STATE_DESC,IKKD_COR_STATE),
                NVL(DECODE(IKKD_BANK_TYPE,'I',UPPER(MAR_FIRST_NAME || ' ' || MAR_MIDDLE_NAME || ' ' || MAR_LAST_NAME),
'O',IKKD_OTH_BANK_CUST_NAME),'NAME'),          /_Ver 3.1/
                NVL(IKKD_BANK_TYPE,'I'),                                          /Ver 3.2/
                NVL(IKKD_PENNY_REQUEST,0)                                          /Ver 3.2/
        INTO :sql_ikkd_name:ind_ikkd_name,
   :sql_ikkd_cor_add1:ind_ikkd_cor_add1,
  :sql_ikkd_cor_add2:ind_ikkd_cor_add2,
  :sql_ikkd_cor_add3:ind_ikkd_cor_add3,
  :sql_ikkd_cor_city:ind_ikkd_cor_city,
  :sql_ikkd_cor_ctry:ind_ikkd_cor_ctry,
  :sql_ikkd_cor_pincd:ind_ikkd_cor_pincd,
  :sql_ikkd_cor_state:ind_ikkd_cor_state,
                :vc_bank_name:ind_bnk_nm,                                          /Ver 3.1/
                :c_bank_type,                                                      /Ver 3.2/
                :i_penny_cnt_ikkd                                                  /Ver 3.2/
       FROM IKKD_INFO_KRA_KYC_DTLS,
  MAR_MBL_ACCOPN_RQST,
  ICM_INFO_CITY_MASTER,
  ISM_INFO_STATE_MASTER,
  ICM_INFO_COUNTRY_MASTER
WHERE   IKKD_PAN_NO = :sql_mar_pan_no
AND   MAR_PAN_NO = IKKD_PAN_NO
AND   MAR_ACCOPN_SRC = IKKD_ACCOPN_SOURCE
AND   MAR_STATUS = 'Z'
AND   IKKD_ACCOPN_SOURCE = 'E'
AND   IKKD_KRA_ACTDEACT = 'A'
      AND ICM_ACTIVE_FLG(+) = 'Y'
      AND ISM_ACTIVE_FLAG(+) = 'Y'
      AND ICM_STATE_CD(+) = ISM_STATE_CD /*** Ver 2.2 ***/
AND   IKKD_COR_CITY = ICM_CITY_DESC(+)
AND   IKKD_COR_STATE = ISM_KRA_MAP_CD(+)
AND   IKKD_COR_CTRY = ICM_KRA_MAP_CD(+);  

    }
    else if(strcmp(vc_address_source.arr,"DIGILOCKER") 0 || strcmp(vc_address_source.arr,"OFFLINEKYC") 0)
    {
    EXEC SQL
      SELECT   TRIM(IKKD_APP_FIRST_NAME||' '||IKKD_APP_MIDDLE_NAME||' '||IKKD_APP_LAST_NAME),
               IKKD_APP_ADDRESS_1,
              IKKD_APP_ADDRESS_2,
              IKKD_APP_ADDRESS_3,
              IKKD_APP_CITY,
             IKKD_APP_COUNTRY,
              IKKD_APP_PINCODE,
              IKKD_APP_STATE,
              NVL(MAR_ADDRESS_USED_FROM,'DIGILOCKER'),
              NVL(DECODE(IKKD_BANK_TYPE,'I',UPPER(MAR_FIRST_NAME || ' ' || MAR_MIDDLE_NAME || ' ' || MAR_LAST_NAME),
'O',IKKD_OTH_BANK_CUST_NAME),'NAME'), /Ver 3.1/
              NVL(IKKD_BANK_TYPE,'I'),   /Ver 3.2/
NVL(IKKD_PENNY_REQUEST,0)   /Ver 3.2/
      INTO :sql_ikkd_name:ind_ikkd_name,
              :sql_ikkd_cor_add1:ind_ikkd_cor_add1,
              :sql_ikkd_cor_add2:ind_ikkd_cor_add2,
              :sql_ikkd_cor_add3:ind_ikkd_cor_add3,
              :sql_ikkd_cor_city:ind_ikkd_cor_city,
              :sql_ikkd_cor_ctry:ind_ikkd_cor_ctry,
              :sql_ikkd_cor_pincd:ind_ikkd_cor_pincd,
              :sql_ikkd_cor_state:ind_ikkd_cor_state,
              :vc_address_source,
              :vc_bank_name:ind_bnk_nm, /Ver 3.1/
              :c_bank_type, /Ver 3.2/
:i_penny_cnt_ikkd /Ver 3.2/
      FROM IKKD_INFO_KRA_KYC_DTLS,
              MAR_MBL_ACCOPN_RQST,
             ICM_INFO_CITY_MASTER,
ISM_INFO_STATE_MASTER,
ICM_INFO_COUNTRY_MASTER
      WHERE IKKD_PAN_NO = :sql_mar_pan_no
      AND MAR_PAN_NO = IKKD_PAN_NO
      AND MAR_ACCOPN_SRC = IKKD_ACCOPN_SOURCE
      AND MAR_STATUS = 'Z'
      /** AND MAR_KRA_ELIGIBLE = 'Y' **/
      AND IKKD_ACCOPN_SOURCE = 'E'
AND IKKD_KRA_ACTDEACT = 'A'
      AND IKKD_COR_CITY = ICM_CITY_DESC(+)
AND IKKD_COR_STATE = ISM_KRA_MAP_CD(+)
AND IKKD_COR_CTRY = ICM_KRA_MAP_CD(+)
      AND ICM_STATE_CD = ISM_STATE_CD     /**Added in Ver 9.3 /
      AND ISM_ACTIVE_FLAG(+) = 'Y'; / Ver 4.2 **/
    }
    /***Ver 3.1 starts ************/
    else
    {
      EXEC SQL
  SELECT  NVL(DECODE(IKKD_BANK_TYPE,'I',UPPER(MAR_FIRST_NAME || ' ' || MAR_MIDDLE_NAME || ' ' || MAR_LAST_NAME),
'O',IKKD_OTH_BANK_CUST_NAME),'NAME'),
                NVL(IKKD_BANK_TYPE,'I'), /Ver 3.2/
  NVL(IKKD_PENNY_REQUEST,0) /Ver 3.2/
        INTO    :vc_bank_name:ind_bnk_nm,
                :c_bank_type, /Ver 3.2/
  :i_penny_cnt_ikkd /Ver 3.2/
        FROM IKKD_INFO_KRA_KYC_DTLS,
  MAR_MBL_ACCOPN_RQST
        WHERE IKKD_PAN_NO = :sql_mar_pan_no
        AND MAR_PAN_NO = IKKD_PAN_NO
  AND MAR_ACCOPN_SRC = IKKD_ACCOPN_SOURCE
  AND MAR_STATUS = 'Z'
        AND IKKD_ACCOPN_SOURCE = 'E'
  AND IKKD_KRA_ACTDEACT = 'A';
    }
    /***Ver 3.1 ends **************/

    if(SQLCODE != 0 && SQLCODE != NO_DATA_FOUND)
{
tpfree((char )ptr_fml_Obuffer);
errlog(c_ServiceName, "S31440",SQLMSG,(char )DEF_USR,DEF_SSSN, c_err_msg);
Fadd32(ptr_fml_Ibuffer, FML_ERR_MSG, c_err_msg, 0) ;
tpreturn(TPFAIL, 0, (char *)ptr_fml_Ibuffer, 0L, 0) ;
}
    if(SQLCODE == NO_DATA_FOUND)
    {
      strcpy(vc_address_source.arr,"KRA");
      SETLEN(vc_address_source);
    }
    if(strlen(sql_ikkd_name.arr) == 0)
    {
      strcpy(vc_address_source.arr,"KRA");
SETLEN(vc_address_source);
    }
SETNULL(sql_ikkd_name);
SETNULL(sql_ikkd_cor_add1);
SETNULL(sql_ikkd_cor_add2);
SETNULL(sql_ikkd_cor_add3);
SETNULL(sql_ikkd_cor_city);
SETNULL(sql_ikkd_cor_ctry);
SETNULL(sql_ikkd_cor_pincd);
SETNULL(sql_ikkd_cor_state);
    SETNULL(vc_address_source);
    SETNULL(vc_bank_name);                      /Ver 3.1/
    userlog("sql_ikkd_name :%s:",sql_ikkd_name.arr);
    userlog("sql_ikkd_cor_add1 :%s:",sql_ikkd_cor_add1.arr);
    userlog("sql_ikkd_cor_add2 :%s:",sql_ikkd_cor_add2.arr);
    userlog("sql_ikkd_cor_add3 :%s:",sql_ikkd_cor_add3.arr);
    userlog("sql_ikkd_cor_city :%s:",sql_ikkd_cor_city.arr);
    userlog("sql_ikkd_cor_ctry :%s:",sql_ikkd_cor_ctry.arr);
    userlog("sql_ikkd_cor_pincd :%s:",sql_ikkd_cor_state.arr);
    userlog("vc_address_source :%s:",vc_address_source.arr);
    userlog("vc_bank_name :%s:",vc_bank_name.arr);              /Ver 3.1/
    userlog("c_bank_type :%c:",c_bank_type);                    /Ver 3.2/
    userlog("i_penny_cnt_ikkd :%d:",i_penny_cnt_ikkd);          /Ver 3.2/

    /Ver 3.2 starts********/

    i_penny_cnt_par = 0;  
    c_par_oao_digi_flow = 'Y'; /ver 9.2/
    EXEC SQL
SELECT PAR_OAO_PENNY_MAX_COUNTER,NVL(APM_OAO_DIGI_FLOW,'Y')  /ver 9.2/
INTO :i_penny_cnt_par,:c_par_oao_digi_flow /ver 9.2/
FROM PAR_SYSTM_PRMTR,APM_ACCOPN_PARAM_MSTR;      /ver 9.2/
    if(SQLCODE != 0)
{
tpfree((char )ptr_fml_Obuffer);
errlog(c_ServiceName, "S31445",SQLMSG,(char )DEF_USR,DEF_SSSN, c_err_msg);
Fadd32(ptr_fml_Ibuffer, FML_ERR_MSG, c_err_msg, 0) ;
tpreturn(TPFAIL, 0, (char _)ptr_fml_Ibuffer, 0L, 0) ;
}

    userlog("i_penny_cnt_par  :%d:",i_penny_cnt_par);
    userlog("c_par_oao_digi_flow :%c", c_par_oao_digi_flow);    /_ver 9.2*/
     /ver 9.2 starts/
if(c_par_oao_digi_flow 'N')
{
if( (strcmp(vc_address_source.arr,"KRA") 0 && c_kra_eligible 'N') ||
(strcmp(vc_address_source.arr,"DIGILOCKER") 0) ||
(strcmp(vc_address_source.arr,"OFFLINEKYC") == 0) )
{
strcpy(vc_address_source.arr,"OFFLINEKYC");
SETLEN(vc_address_source);
}
}
userlog("vc_address_source |%s|",vc_address_source.arr);
/ver 9.2 ends/
    /Ver 3.2 ends/  
    INIT(i_err,TOTAL_FML);
INIT(i_err_msg,TOTAL_FML);
i_err[0] = Fadd32(ptr_fml_Obuffer,FML_USR_USR_NM,(char)sql_ikkd_name.arr,0);
i_err_msg[0]=Ferror32;
i_err[1] = Fadd32(ptr_fml_Obuffer,FML_USR_ADDRSS_LN1,(char)sql_ikkd_cor_add1.arr,0);
i_err_msg[1]=Ferror32;
i_err[2] = Fadd32(ptr_fml_Obuffer,FML_USR_ADDRSS_LN2,(char)sql_ikkd_cor_add2.arr,0);
i_err_msg[2]=Ferror32;
i_err[3] = Fadd32(ptr_fml_Obuffer,FML_USR_ADDRSS_LN3,(char)sql_ikkd_cor_add3.arr,0);
i_err_msg[3]=Ferror32;
i_err[4] = Fadd32(ptr_fml_Obuffer,FML_USR_ADDRSS_CTY,(char)sql_ikkd_cor_city.arr,0);
i_err_msg[4]=Ferror32;
    i_err[5] = Fadd32(ptr_fml_Obuffer,FML_USR_ZIP_CD2,(char)sql_ikkd_cor_pincd.arr,0);
i_err_msg[5]=Ferror32;
i_err[6] = Fadd32(ptr_fml_Obuffer,FML_USR_ADDRSS_STTE,(char)sql_ikkd_cor_state.arr,0);
i_err_msg[6]=Ferror32;
i_err[7] = Fadd32(ptr_fml_Obuffer,FML_USR_ADDRSS_CNTRY,(char)sql_ikkd_cor_ctry.arr,0);
i_err_msg[7]=Ferror32;
i_err[8] = Fadd32(ptr_fml_Obuffer,FML_TRANS_TYP,(char)vc_address_source.arr,0);
i_err_msg[8]=Ferror32;
    i_err[9] = Fadd32(ptr_fml_Obuffer,FML_MMD_CHNG,(char*)vc_bank_name.arr,0);    /Ver 3.1/
i_err_msg[9]=Ferror32;                                                              /Ver 3.1/
    i_err[10] = Fadd32(ptr_fml_Obuffer,FML_OTRNSCTN_FLW,(char*)&c_bank_type,0);            /Ver 3.2/
i_err_msg[10]=Ferror32; /Ver 3.2/
    i_err[11] = Fadd32(ptr_fml_Obuffer,FML_PRTFLO_ID,(char*)&i_penny_cnt_ikkd,0);       /Ver 3.2/
i_err_msg[11]=Ferror32; /Ver 3.2/
    i_err[12] = Fadd32(ptr_fml_Obuffer,FML_TRN_NO,(char*)&i_penny_cnt_par,0);      /Ver 3.2/
i_err_msg[12]=Ferror32; /Ver 3.2/
    /* Ver 3.9 starts ***/
i_err[13] = Fadd32( ptr_fml_Obuffer,FML_NRE_FLG,(char*)vc_stage_stts.arr, 0 );
i_err_msg[13]=Ferror32;
/* Ver 3.9 ends *****/
    for(i_err_cnt=0; i_err_cnt < 14; i_err_cnt++)
{
if(i_err[i_err_cnt]==-1)
{
tpfree((char )ptr_fml_Obuffer);
userlog("Error in Fget for FML :%d:", i_err_cnt);
errlog(c_ServiceName, "S31450",Fstrerror32(i_err_msg[i_err_cnt]),(char )DEF_USR,DEF_SSSN, c_err_msg);
Fadd32(ptr_fml_Ibuffer, FML_ERR_MSG, c_err_msg, 0) ;
tpreturn(TPFAIL, 0, (char *)ptr_fml_Ibuffer, 0L, 0) ;
}
}