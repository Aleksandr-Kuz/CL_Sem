#encoding "utf-8"
#GRAMMAR_ROOT S


Name -> Word<gram="persn">;
Surname -> Word<gram="famn">;
Patronymic -> Word<gram="patrn">;


initial -> Word<wfm=/([А-Я]{1}\s*\.)/>;
initials -> initial interp(PersonFact.name) initial interp(PersonFact.patrn);

// И. И. Иванов
S -> initials Surname interp(PersonFact.surname);

// И. Иванов
S -> initial interp(PersonFact.name) Surname interp(PersonFact.surname);

// Иванов И. И.
S -> Surname interp(PersonFact.surname) initials;

// Иванов И.
S -> Surname interp(PersonFact.surname) initial interp(PersonFact.name);

// Иванов
S -> Surname interp(PersonFact.surname);

// Иван Иванов
S -> Name interp(PersonFact.name)
    Surname interp(PersonFact.surname);

// Иван Иванович
S -> Name interp(PersonFact.name)
    Patronymic interp(PersonFact.patrn);