'''
The functions below were used to code the language and domain of the news articles.

'''



def code_lang(df):
    pattern1 = r"RU"
    pattern2 = r"HI"
    pattern3 = r"PT"
    pattern4 = r"BG"
    pattern5 = r"EN"

    for index, row in df.iterrows():
        if re.search(pattern1, row['Filename']):
            df.loc[index, 'Language'] = 'RU'
        elif re.search(pattern2, row['Filename']):
            df.loc[index, 'Language'] = 'HI'
        elif re.search(pattern3, row['Filename']):
            df.loc[index, 'Language'] = 'PT'
        elif re.search(pattern4, row['Filename']):
            df.loc[index, 'Language'] = 'BG'
        elif re.search(pattern5, row['Filename']):
            df.loc[index, 'Language'] = 'EN'
        else:
            df.loc[index, 'Language'] = '-'
    return df


def code_domain_filename(df):
    pattern1 = r"CC"
    pattern2 = r"UA|RU|URW"

    for index, row in df.iterrows():
        if re.search(pattern1, row['Filename']):
            df.loc[index, 'Domain'] = 'CC'
        elif re.search(pattern2, row['Filename']):
            df.loc[index, 'Domain'] = 'UA'
        else:
            df.loc[index, 'Domain'] = '-'
    return df


def code_domain_text(df):
    cc_keywords = r"\b(climate|warming|carbon|greenhouse|renewable|sustainability|heat|deforestation|weather|flooding)\b"
    ua_keywords = r"\b(Ukraine|Russia|NATO|conflict|sanctions|Putin|Zelensky|war|invasion|Russian|Ukrainian|Lugansk|enemy)\b"

    for index, row in df[df['Domain']=='-'].iterrows():
        if re.search(cc_keywords, row['Translated_Text'],re.IGNORECASE):
            df.loc[index, 'Domain'] = 'CC'
        elif re.search(ua_keywords, row['Translated_Text'],re.IGNORECASE):
            df.loc[index, 'Domain'] = 'UA'
        else:
            df.loc[index, 'Domain'] = '-'
    return df


