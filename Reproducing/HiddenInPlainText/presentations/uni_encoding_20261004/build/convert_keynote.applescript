with timeout of 120 seconds
    tell application "Keynote"
        set deck to open POSIX file "/Users/kirilbangachev/Documents/ChatGPT/Steganography/presentations/uni_encoding_20261004/output/University_Encoding.pptx"
        set slideCount to count of slides of deck
        if slideCount is not 5 then error "Unexpected slide count"
        save deck in POSIX file "/Users/kirilbangachev/Documents/ChatGPT/Steganography/presentations/uni_encoding_20261004/output/University_Encoding.key"
        export deck to POSIX file "/Users/kirilbangachev/Documents/ChatGPT/Steganography/presentations/uni_encoding_20261004/output/University_Encoding.pdf" as PDF with properties {PDF image quality:Best}
        return {name of deck, slideCount, count of charts of slide 2 of deck, count of tables of slide 3 of deck, count of tables of slide 4 of deck, count of tables of slide 5 of deck}
    end tell
end timeout
