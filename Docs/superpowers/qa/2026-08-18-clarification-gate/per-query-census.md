# Per-query census — all 60 golden queries (TASK-16072 AC#1)

Produced by `census.py` in this directory. `rel` = labelled relevant docs;
`match` = corpus docs containing every content word; `alt` = match − rel, i.e.
readings the corpus holds that the query does not distinguish.

```
PROBE PROOF: docs with non-empty text: 172 of 172
query                                          category             rel match alt  gate?
which outstation does not take a standard mai  negation               1     3    3  YES
Zephyr-9 flywheel assembly balance tolerance   keyword                1     1    0  -
asset tag QX-8842                              keyword                1     1    0  -
Halcyon-4 ledger reconciliation batch abort    keyword                1     1    0  -
Marlstone kiln refractory lining               keyword                1     1    0  -
Nimbus-14 firmware rollback                    keyword                2     1    0  -
Calyx-77 torque limiter slipping               keyword                2     0    0  -
Obsidian-3 lathe spindle bearing               keyword                1     1    0  -
Quillon-6 antenna mast guy tension             keyword                1     1    0  -
Verdigris-8 anti-corrosion coating salt cabin  keyword                1     1    0  -
Pellucid-12 vacuum gauge calibration           keyword                1     1    0  -
Thimble-5 relay board swap                     keyword                1     1    0  -
Ashgrove-2 coolant pump seal                   keyword                1     1    0  -
Fennimore-3 packaging line changeover accepta  keyword                1     1    0  -
Larkspur-11 turbine commissioning walkthrough  keyword                1     1    0  -
Drayton-6 conveyor belt tracking               keyword                1     1    0  -
plant maintenance record                       keyword                1     1    0  -
outstations that are not reached by a surface  negation               1     0    0  -
which mast does not carry a standard three-pa  negation               1     0    0  -
sourdough starter hydration ratio              negative               0     0    0  -
visa paperwork for antarctic research bases    negative               0     0    0  -
tuning a mandolin by ear                       negative               0     0    0  -
olympic swimming pool lane dimensions          negative               0     0    0  -
medieval castle siege tactics                  negative               0     0    0  -
why honeybee colonies collapse                 negative               0     0    0  -
chess endgame tablebases                       negative               0     0    0  -
yearly sales increased sharply                 paraphrase             1     0    0  -
the warehouse relocated into a bigger buildin  paraphrase             1     0    0  -
the project deadline was delayed by roughly a  paraphrase             1     0    0  -
hiring was stopped across the company          paraphrase             1     0    0  -
the plane landed without injury after the mot  paraphrase             1     0    0  -
a lengthy drought reduced corn yields          paraphrase             1     0    0  -
thieves took artwork from the museum after da  paraphrase             1     0    0  -
the bridge shut for road repaving              paraphrase             1     0    0  -
a solar array installed to lower power bills   paraphrase             1     0    0  -
a buyer wanted money returned because the pac  paraphrase             1     0    0  -
the daily sync moved to later in the week      paraphrase             1     0    0  -
the website was unreachable after an electric  paraphrase             1     0    0  -
moved my exercise sessions to sunrise because  paraphrase             1     0    0  -
prompt that turns a shift log into a short su  prompt                 1     0    0  -
template for building an incident timeline fr  prompt                 1     0    0  -
saved prompt for chasing a supplier about a l  prompt                 1     0    0  -
prompt that pulls the actions out of meeting   prompt                 1     0    0  -
prompt that builds a glossary of terms from a  prompt                 1     0    0  -
pump chamber inspection                        scoped                 1     1    0  -
storm overflow record                          scoped                 1     1    0  -
intake screen survey                           scoped                 1     1    0  -
meter box key                                  scoped                 1     1    0  -
valve pit access                               scoped                 1     1    0  -
sample point sign                              scoped                 1     1    0  -
duty board notice                              scoped                 1     1    0  -
warning signs of a heart attack                vocabulary_mismatch    1     0    0  -
how are kidney stones removed                  vocabulary_mismatch    1     0    0  -
what the tail of an airplane does              vocabulary_mismatch    1     0    0  -
how strong was the earthquake                  vocabulary_mismatch    1     0    0  -
how dangerous is high blood pressure           vocabulary_mismatch    1     0    0  -
who inherits when someone leaves no will       vocabulary_mismatch    1     0    0  -
gum disease treatment                          vocabulary_mismatch    1     0    0  -
which painkiller is easier on the stomach      vocabulary_mismatch    1     0    0  -
does nearsightedness worsen in children        vocabulary_mismatch    1     0    0  -
QUALIFYING (corpus holds an unlabelled alternative reading): 1 of 60
   'which outstation does not take a standard mains supply' [negation]
```
