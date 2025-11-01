# Search and classification of ornaments in 16th century lute tablature

TODO: add abstract

------------------------------------------------------------------

## search_for_rules.py

TODO: abrupt change in rhythm (change from e.g.: 1/4 to 1/16), show onsets where this is happening (if at all?)


TODO: fix rests before or after groups (use onset time??)


TODO: is it possible to infer rests within a voice with onset and duration info?
add that to dataframe(s)

TODO: do not use "ornamentation"-flag as starting point but use groups of short consecutive notes as a starting point??


TODO: look up meter in the tbp-file or mei


TODO: make version that uses squished voices and chord context (that's probably the way to go)


TODO: flag weird stuff:
  - 16th that does not start on a downbeat ("starts with a rest")
      - See screenshot: first note is in the wrong voice (Ach unfall Sopran/Alt Measure 7)
  - group of ornaments starts or ends on a dissonance

extract all the ornament-groups (notes tagged as ornamentations and add note before and note after (see other notes))
and then check, if groups possibly need to be merged, because they are separated by a not annotated note -> based on onset and duration

