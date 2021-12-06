parser Countermeaures:
    ignore:           '\\s+'
    token INUM:       '[0-9]+'
    token FNUM:       '[0-9]*\\.[0-9]+'
    token JITTER:     'ClockJitter'
    token DESYNC:     'Desync'
    token RDI:        'RDI'
    token UNIFORM:    'UniformNoise'

    rule countermeasure: jitter   {{return jitter}}
                        | desync  {{return desync}}
                        | rdi     {{return rdi}}
                        | uniform {{return uniform}}

    rule jitter: JITTER   {{ result = ['jitter'] }}
                 "\\(jitters_level=" INUM "\\)"  {{ return result + [ int(INUM) ] }}

    rule desync: DESYNC   {{ result = ['desync'] }}
                 "\\(desync_level=" INUM "\\)"  {{ return result + [ int(INUM) ] }}

    rule rdi: RDI   {{ result = ['rdi'] }}
                 "\\(A=" INUM  {{ result.append(int(INUM)) }}
                 ",B=" INUM  {{ result.append(int(INUM)) }}
                 ",probability=" FNUM  {{ result.append(float(FNUM)) }}
                 ",amplitude=" FNUM "\\)" {{ return result + [ float(FNUM) ] }}

    rule uniform: UNIFORM   {{ result = ['uniform'] }}
                 "\\(noise_factor=" FNUM   {{ result.append(float(FNUM)) }}
                 ",noise_scale=" FNUM "\\)"  {{ return result + [ float(FNUM) ] }}

    rule countermeasures: "\\["               {{ result = [] }}
                          (countermeasure    {{ result.append(countermeasure)  }} )*
                          ("," countermeasure {{ result.append(countermeasure) }} )*
                          "\\]"               {{ return result }}
