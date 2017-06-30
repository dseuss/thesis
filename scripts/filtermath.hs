#!/usr/bin/env stack exec runhaskell
-- filtermath.hs
import Text.Pandoc.JSON

main :: IO ()
main = toJSONFilter filterMath
  where filterMath (Math _ _) = []
        filterMath x = [x]
